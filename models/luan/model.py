import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def get_2d_normalized_sincos_pos_embed(embed_dim, h, w, device):
    """
    解像度やアスペクト比が変化しても破綻しないよう、
    座標を [-1.0, 1.0] に正規化してから生成する2D位置エンコーディング
    """
    # アスペクト比を反映させるため、長辺を基準にスケーリングするか、[-1, 1] の相対座標を付与
    grid_h = torch.linspace(-1.0, 1.0, steps=h, device=device) * (h / max(h, w)) * 16.0
    grid_w = torch.linspace(-1.0, 1.0, steps=w, device=device) * (w / max(h, w)) * 16.0
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

    emb_h = get_timestep_embedding(grid_h.flatten(), embed_dim // 2)
    emb_w = get_timestep_embedding(grid_w.flatten(), embed_dim // 2)
    pos_embed = torch.cat([emb_h, emb_w], dim=1).unsqueeze(0)  # [1, h*w, embed_dim]
    return pos_embed


class ConvNeXtBlock(nn.Module):
    """Pre-Norm と Depthwise Conv を組み合わせた本番用残差ブロック"""

    def __init__(self, dim, kernel_size=5, num_groups=8):
        super().__init__()
        padding = kernel_size // 2
        groups = min(num_groups, dim)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )
        self.pwconv1 = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class TripleStreamModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dim_A=128,
        hidden_dim_B=64,
        hidden_dim_C=64,
        text_dim=128,
    ):
        super().__init__()

        # --- 1. 解像度別プーリング ---
        self.pool_A = nn.AvgPool2d(kernel_size=4, stride=4)  # H/4, W/4
        self.pool_B = nn.AvgPool2d(kernel_size=2, stride=2)  # H/2, W/2

        self.time_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.GELU(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )

        self.t_proj_A = nn.Linear(self.time_dim, hidden_dim_A)
        self.t_proj_B = nn.Linear(self.time_dim, hidden_dim_B)
        self.t_proj_C = nn.Linear(self.time_dim, hidden_dim_C)

        # --- Stream A (Global / 低周波) ---
        self.proj_A = nn.Conv2d(in_channels, hidden_dim_A, 1)
        self.norm_A = nn.LayerNorm(hidden_dim_A)
        self.attn_A = nn.MultiheadAttention(
            embed_dim=hidden_dim_A, num_heads=8, batch_first=True
        )
        self.text_proj_A = (
            nn.Linear(text_dim, hidden_dim_A)
            if text_dim != hidden_dim_A
            else nn.Identity()
        )

        # --- Stream B (Relational / 中周波) ---
        self.proj_B = nn.Conv2d(in_channels, hidden_dim_B, 1)
        # 5x5 受容野を持つ2層残差ミキサー（勾配監視用に mix_B[0].weight を先頭に配置）
        self.mix_B = nn.Sequential(
            nn.Conv2d(
                hidden_dim_B,
                hidden_dim_B,
                kernel_size=5,
                padding=2,
                groups=hidden_dim_B,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim_B, hidden_dim_B, 1),
            ConvNeXtBlock(hidden_dim_B, kernel_size=5, num_groups=8),
        )

        # --- Stream C (Texture / 高周波) ---
        self.proj_C = nn.Conv2d(in_channels, hidden_dim_C, 1)
        self.norm_C_attn = nn.LayerNorm(hidden_dim_C)
        self.cross_attn_C = nn.MultiheadAttention(
            embed_dim=hidden_dim_C, num_heads=4, batch_first=True
        )
        self.text_proj_C = nn.Linear(text_dim, hidden_dim_C)

        # Pre-Norm 構造の2層テクスチャ抽出ブロック（スジ状ノイズを解消）
        self.conv_C = nn.Sequential(
            ConvNeXtBlock(hidden_dim_C, kernel_size=3, num_groups=8),
            ConvNeXtBlock(hidden_dim_C, kernel_size=3, num_groups=8),
        )

        # --- Fusion層 (カスケード型アフィン変換) ---
        self.fusion_A_to_B = nn.Conv2d(
            hidden_dim_A, hidden_dim_B, kernel_size=3, padding=1
        )
        self.fusion_refine_AB = nn.Sequential(
            nn.GroupNorm(8, hidden_dim_B),
            nn.SiLU(),
            nn.Conv2d(hidden_dim_B, hidden_dim_B, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.fusion_AB_to_C_gamma = nn.Conv2d(
            hidden_dim_B, hidden_dim_C, kernel_size=3, padding=1
        )
        self.fusion_AB_to_C_beta = nn.Conv2d(
            hidden_dim_B, hidden_dim_C, kernel_size=3, padding=1
        )

        # 最終出力層
        self.out_norm = nn.GroupNorm(
            num_groups=min(8, hidden_dim_C), num_channels=hidden_dim_C
        )
        self.out_act = nn.GELU()
        self.out_proj = nn.Conv2d(hidden_dim_C, in_channels, kernel_size=3, padding=1)

    def forward(self, x, context, t):
        """
        x: [B, in_channels, H, W] (任意の解像度・アスペクト比)
        context: [B, M, text_dim] (フルサイズのテキスト特徴量)
        t: [B]
        """
        B, C, H, W = x.shape

        t_emb = get_timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        t_A = self.t_proj_A(t_emb).unsqueeze(1)  # [B, 1, C_A]
        t_B = self.t_proj_B(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C_B, 1, 1]
        t_C = self.t_proj_C(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C_C, 1, 1]

        # === 1. Split (解像度分割) ===
        x_A = self.pool_A(x)
        x_B = self.pool_B(x)
        x_C = x

        # === 2. Stream A (Global) ===
        h_A = self.proj_A(x_A)
        _, C_A, H_A, W_A = h_A.shape
        h_A_flat = h_A.view(B, C_A, -1).permute(0, 2, 1)  # [B, H_A*W_A, C_A]

        pos_embed_A = get_2d_normalized_sincos_pos_embed(C_A, H_A, W_A, x.device)
        h_A_flat = h_A_flat + pos_embed_A

        ctx_A = self.text_proj_A(context)
        seq_len_A = h_A_flat.shape[1]
        concat_input = torch.cat([h_A_flat, ctx_A, t_A], dim=1)
        concat_norm = self.norm_A(concat_input)

        attn_out_flat, _ = self.attn_A(concat_norm, concat_norm, concat_norm)
        out_A_flat = h_A_flat + attn_out_flat[:, :seq_len_A, :]
        out_A = out_A_flat.permute(0, 2, 1).view(B, C_A, H_A, W_A)

        # === 3. Stream B (Relational) ===
        h_B = self.proj_B(x_B) + t_B
        out_B = h_B + self.mix_B(h_B)

        # === 4. Stream C (Texture) ===
        h_C = self.proj_C(x_C) + t_C
        _, C_C, H_C, W_C = h_C.shape
        h_C_flat = h_C.view(B, C_C, -1).permute(0, 2, 1)  # [B, H_C*W_C, C_C]

        # Stream C にも正規化2D位置エンコーディングを付与してスジ状アーティファクトを防止
        pos_embed_C = get_2d_normalized_sincos_pos_embed(C_C, H_C, W_C, x.device)
        h_C_q = self.norm_C_attn(h_C_flat + pos_embed_C)

        # 4トークンへの強制プーリングを廃止し、全文コンテキストを直接参照
        ctx_proj = self.text_proj_C(context)
        attn_out_C, _ = self.cross_attn_C(h_C_q, ctx_proj, ctx_proj)
        h_C_flat = h_C_flat + attn_out_C
        h_C_attn = h_C_flat.permute(0, 2, 1).view(B, C_C, H_C, W_C)

        # Pre-Norm 畳み込みブロック（出力の振幅を潰さずに Fusion へ渡す）
        out_C = self.conv_C(h_C_attn)

        # === 5. Fusion層 (カスケード型アフィン変換) ===
        """
        up_A = F.interpolate(
            out_A,
            size=(out_B.shape[2], out_B.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        fusion_AB = self.fusion_A_to_B(up_A) + out_B

        up_AB = F.interpolate(
            fusion_AB, size=(H, W), mode="bilinear", align_corners=False
        )
        gamma = self.fusion_AB_to_C_gamma(up_AB)
        beta = self.fusion_AB_to_C_beta(up_AB)
        """

        up_A = F.interpolate(
            out_A,
            size=(out_B.shape[2], out_B.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        fusion_AB = self.fusion_A_to_B(up_A) + out_B

        up_AB = F.interpolate(
            fusion_AB, size=(H, W), mode="bilinear", align_corners=False
        )
        # 拡大された特徴マップのボケを 3x3 Conv で引き締めてから gamma / beta を計算
        up_AB = up_AB + self.fusion_refine_AB(up_AB)
        gamma = self.fusion_AB_to_C_gamma(up_AB)
        beta = self.fusion_AB_to_C_beta(up_AB)

        fusion_final = (1.0 + gamma) * out_C + beta

        # === 6. Output ===
        output = self.out_proj(self.out_act(self.out_norm(fusion_final)))
        return output


# 既存コードとの互換エイリアス
TripleStreamPoC = TripleStreamModel
