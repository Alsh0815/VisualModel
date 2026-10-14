import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TripleStreamPoC(nn.Module):
    def __init__(
        self, in_channels=4, hidden_dim_A=128, hidden_dim_B=64, hidden_dim_C=16
    ):
        super().__init__()

        # --- 1. 解像度別プーリング (入力ノイズのダウンサンプル) ---
        self.pool_A = nn.AvgPool2d(kernel_size=4, stride=4)  # 64x64 -> 16x16
        self.pool_B = nn.AvgPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        # Stream C はプーリングなし (64x64維持)

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
        self.attn_A = nn.MultiheadAttention(
            embed_dim=hidden_dim_A, num_heads=8, batch_first=True
        )

        # --- Stream B (Relational / 中周波) ---
        self.proj_B = nn.Conv2d(in_channels, hidden_dim_B, 1)
        # PoC用簡易SSM/RWKVの代用として軽量なDW-Conv + 1x1 Convミキサー
        self.mix_B = nn.Sequential(
            nn.Conv2d(
                hidden_dim_B,
                hidden_dim_B,
                kernel_size=3,
                padding=1,
                groups=hidden_dim_B,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim_B, hidden_dim_B, 1),
        )

        # --- Stream C (Texture / 高周波) ---
        self.proj_C = nn.Conv2d(in_channels, hidden_dim_C, 1)
        self.conv_C = nn.Sequential(
            nn.Conv2d(
                hidden_dim_C,
                hidden_dim_C,
                kernel_size=3,
                padding=1,
                groups=hidden_dim_C,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim_C, hidden_dim_C, 1),
        )
        self.norm_C = nn.GroupNorm(num_groups=4, num_channels=hidden_dim_C)

        # --- Cross-Attention (Stream C 用) ---
        self.cross_attn_C = nn.MultiheadAttention(
            embed_dim=hidden_dim_C, num_heads=1, batch_first=True
        )
        # プロンプト(128次元)をStream Cの次元に合わせる
        self.text_proj = nn.Linear(128, hidden_dim_C)

        # --- Fusion層 (カスケード型アフィン変換) ---
        # A -> B 結合用
        self.fusion_A_to_B = nn.Conv2d(hidden_dim_A, hidden_dim_B, 1)
        # A+B -> C 変調用パラメータ生成 (γ, β)
        self.fusion_AB_to_C_gamma = nn.Conv2d(hidden_dim_B, hidden_dim_C, 1)
        self.fusion_AB_to_C_beta = nn.Conv2d(hidden_dim_B, hidden_dim_C, 1)

        # 最終出力用
        self.out_proj = nn.Conv2d(hidden_dim_C, in_channels, 1)

    def forward(self, x, context, t):
        """
        x: [B, in_channels, H, W] (潜在ノイズ, 例: 64x64)
        context: [B, 1, 128] (テキストエンコーダ出力, M=1に圧縮済み)
        """
        B, C, H, W = x.shape

        t_emb = get_timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # 各ストリームの次元に合わせたタイムステップベクトル
        t_A = self.t_proj_A(t_emb).unsqueeze(1)  # [B, 1, 128]
        t_B = self.t_proj_B(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        t_C = self.t_proj_C(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, 16, 1, 1]

        # === 1. Split (解像度分割) ===
        x_A = self.pool_A(x)  # [B, C, H/4, W/4]
        x_B = self.pool_B(x)  # [B, C, H/2, W/2]
        x_C = x  # [B, C, H, W]

        # === 2. Stream A (Global) ===
        h_A = self.proj_A(x_A)  # [B, 128, H/4, W/4]
        _, C_A, H_A, W_A = h_A.shape
        # Attention用にFlatten: [B, H_A*W_A, 128]
        h_A_flat = h_A.view(B, C_A, -1).permute(0, 2, 1)

        # 【追加】2D Positional Encodingの生成と加算
        grid_h = torch.arange(H_A, dtype=torch.float32, device=x.device)
        grid_w = torch.arange(W_A, dtype=torch.float32, device=x.device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

        # 縦横それぞれ次元数の半分(C_A // 2)を使ってエンコード
        emb_h = get_timestep_embedding(grid_h.flatten(), C_A // 2)
        emb_w = get_timestep_embedding(grid_w.flatten(), C_A // 2)
        pos_embed_A = torch.cat([emb_h, emb_w], dim=1).unsqueeze(0)  # [1, H_A*W_A, C_A]

        h_A_flat = h_A_flat + pos_embed_A

        # 画像テンソル、テキスト特徴量、タイムステップを結合(Concat)してSelf-Attentionへ
        seq_len_A = h_A_flat.shape[1]
        concat_input = torch.cat([h_A_flat, context, t_A], dim=1)

        attn_out_flat, _ = self.attn_A(concat_input, concat_input, concat_input)

        out_A_flat = h_A_flat + attn_out_flat[:, :seq_len_A, :]
        out_A = out_A_flat.permute(0, 2, 1).view(B, C_A, H_A, W_A)  # [B, 128, H/4, W/4]

        # === 3. Stream B (Relational) ===
        h_B = self.proj_B(x_B) + t_B
        out_B = self.mix_B(h_B) + h_B  # [B, 64, H/2, W/2]

        # === 4. Stream C (Texture) ===
        h_C = self.proj_C(x_C) + t_C
        _, C_C, H_C, W_C = h_C.shape
        h_C_flat = h_C.view(B, C_C, -1).permute(0, 2, 1)

        # 高解像度学習でVRAMが足りない場合は有効化
        # context_C = context.permute(0, 2, 1)  # [B, 128, M]
        # context_C = F.adaptive_avg_pool1d(context_C, 4)  # [B, 128, 4]
        # context_C = context_C.permute(0, 2, 1)  # [B, 4, 128]

        # VRAMが足りない場合は無効化
        context_C = context

        ctx_proj = self.text_proj(context_C)
        attn_out_C, _ = self.cross_attn_C(h_C_flat, ctx_proj, ctx_proj)
        h_C_flat = h_C_flat + attn_out_C
        h_C_attn = h_C_flat.permute(0, 2, 1).view(B, C_C, H_C, W_C)
        out_C = h_C_attn + self.conv_C(self.norm_C(h_C_attn))

        # === 5. Fusion層 (カスケード型アフィン変換) ===
        # Stream A -> B
        up_A = F.interpolate(
            out_A,
            size=(out_B.shape[2], out_B.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        up_A_proj = self.fusion_A_to_B(up_A)
        fusion_AB = up_A_proj + out_B  # [B, 64, H/2, W/2]

        # Stream A+B -> C (Modulation)
        up_AB = F.interpolate(
            fusion_AB, size=(H, W), mode="bilinear", align_corners=False
        )
        gamma = self.fusion_AB_to_C_gamma(up_AB)  # [B, 16, H, W]
        beta = self.fusion_AB_to_C_beta(up_AB)  # [B, 16, H, W]

        # F_final = gamma * Norm(F_C) + beta
        # fusion_final = gamma * out_C + beta  # [B, 16, H, W]
        fusion_final = (1.0 + gamma) * out_C + beta

        # === 6. Output ===
        output = self.out_proj(fusion_final)  # [B, in_channels, H, W]

        return output
