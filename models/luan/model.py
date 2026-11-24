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


def get_2d_normalized_sincos_pos_embed(embed_dim, h, w, device, freq_scale=64.0):
    """
    解像度やアスペクト比が変化しても破綻しない正規化2D位置エンコーディング。
    Stream A は大域用 (freq_scale=64.0)、Stream C は微細位置識別用 (freq_scale=256.0) を使用。
    """
    grid_h = (
        torch.linspace(-1.0, 1.0, steps=h, device=device) * (h / max(h, w)) * freq_scale
    )
    grid_w = (
        torch.linspace(-1.0, 1.0, steps=w, device=device) * (w / max(h, w)) * freq_scale
    )
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

    emb_h = get_timestep_embedding(grid_h.flatten(), embed_dim // 2)
    emb_w = get_timestep_embedding(grid_w.flatten(), embed_dim // 2)
    pos_embed = torch.cat([emb_h, emb_w], dim=1).unsqueeze(0)  # [1, h*w, embed_dim]
    return pos_embed


class ConvNeXtBlock(nn.Module):
    """Pre-Norm + AdaGN(時刻条件変調) + Dilated Depthwise Conv 残差ブロック"""

    def __init__(self, dim, kernel_size=5, dilation=1, num_groups=8, time_dim=None):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        groups = min(num_groups, dim)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
        )
        self.pwconv1 = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.time_mlp = None
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, dim * 2),
            )
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)

    def forward(self, x, t_emb=None):
        residual = x
        x = self.norm(x)
        if self.time_mlp is not None and t_emb is not None:
            scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
            x = x * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(
                -1
            ).unsqueeze(-1)
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class TripleStreamModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dim_A=192,
        hidden_dim_B=96,
        hidden_dim_C=96,
        text_dim=768,
        max_tokens_A_edge=32,
    ):
        super().__init__()

        self.max_tokens_A_edge = max_tokens_A_edge
        self._pos_embed_cache = {}

        self.time_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.GELU(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )

        self.t_proj_A = nn.Linear(self.time_dim, hidden_dim_A)
        self.t_proj_B = nn.Linear(self.time_dim, hidden_dim_B)
        self.t_proj_C = nn.Linear(self.time_dim, hidden_dim_C)

        # --- Stream A (Global / 低周波: 安定した2層 Self-Attention) ---
        self.proj_A = nn.Conv2d(in_channels, hidden_dim_A, kernel_size=3, padding=1)
        self.norm_A = nn.LayerNorm(hidden_dim_A)
        self.attn_A = nn.MultiheadAttention(
            embed_dim=hidden_dim_A, num_heads=8, batch_first=True
        )
        self.ffn_A = nn.Sequential(
            nn.LayerNorm(hidden_dim_A),
            nn.Linear(hidden_dim_A, hidden_dim_A * 2),
            nn.GELU(),
            nn.Linear(hidden_dim_A * 2, hidden_dim_A),
        )
        self.norm_A2 = nn.LayerNorm(hidden_dim_A)
        self.attn_A2 = nn.MultiheadAttention(
            embed_dim=hidden_dim_A, num_heads=8, batch_first=True
        )
        self.ffn_A2 = nn.Sequential(
            nn.LayerNorm(hidden_dim_A),
            nn.Linear(hidden_dim_A, hidden_dim_A * 2),
            nn.GELU(),
            nn.Linear(hidden_dim_A * 2, hidden_dim_A),
        )
        self.spatial_A = ConvNeXtBlock(
            hidden_dim_A, kernel_size=3, num_groups=8, time_dim=self.time_dim
        )
        self.text_proj_A = (
            nn.Linear(text_dim, hidden_dim_A)
            if text_dim != hidden_dim_A
            else nn.Identity()
        )

        # --- Stream B (Relational / 中周波: テキスト条件FiLM + AdaGN Dilation 1,2,4,8) ---
        self.proj_B = nn.Conv2d(in_channels, hidden_dim_B, kernel_size=3, padding=1)
        self.cond_proj_B = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_dim + text_dim, hidden_dim_B * 2),
        )
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
        )
        self.blocks_B = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden_dim_B,
                    kernel_size=5,
                    dilation=1,
                    num_groups=8,
                    time_dim=self.time_dim,
                ),
                ConvNeXtBlock(
                    hidden_dim_B,
                    kernel_size=5,
                    dilation=2,
                    num_groups=8,
                    time_dim=self.time_dim,
                ),
                ConvNeXtBlock(
                    hidden_dim_B,
                    kernel_size=5,
                    dilation=4,
                    num_groups=8,
                    time_dim=self.time_dim,
                ),
                ConvNeXtBlock(
                    hidden_dim_B,
                    kernel_size=5,
                    dilation=8,
                    num_groups=8,
                    time_dim=self.time_dim,
                ),
            ]
        )
        self.post_B = ConvNeXtBlock(
            hidden_dim_B,
            kernel_size=3,
            dilation=1,
            num_groups=8,
            time_dim=self.time_dim,
        )

        # --- Stream C (Texture / 高周波: K/V正規化 Cross-Attention + AdaGN ConvNeXt) ---
        self.proj_C = nn.Conv2d(in_channels, hidden_dim_C, kernel_size=3, padding=1)
        self.pre_attn_ab_to_c = nn.Conv2d(hidden_dim_B, hidden_dim_C, kernel_size=1)
        self.norm_C_attn = nn.LayerNorm(hidden_dim_C)
        self.norm_C_ctx = nn.LayerNorm(hidden_dim_C)
        self.cross_attn_C = nn.MultiheadAttention(
            embed_dim=hidden_dim_C, num_heads=4, batch_first=True
        )
        self.ffn_C = nn.Sequential(
            nn.LayerNorm(hidden_dim_C),
            nn.Linear(hidden_dim_C, hidden_dim_C * 2),
            nn.GELU(),
            nn.Linear(hidden_dim_C * 2, hidden_dim_C),
        )
        self.text_proj_C = nn.Linear(text_dim, hidden_dim_C)

        self.conv_C = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden_dim_C,
                    kernel_size=3,
                    dilation=1,
                    num_groups=8,
                    time_dim=self.time_dim,
                ),
                ConvNeXtBlock(
                    hidden_dim_C,
                    kernel_size=3,
                    dilation=2,
                    num_groups=8,
                    time_dim=self.time_dim,
                ),
                ConvNeXtBlock(
                    hidden_dim_C,
                    kernel_size=3,
                    dilation=4,
                    num_groups=8,
                    time_dim=self.time_dim,
                ),
                ConvNeXtBlock(
                    hidden_dim_C,
                    kernel_size=3,
                    dilation=1,
                    num_groups=8,
                    time_dim=self.time_dim,
                ),
            ]
        )

        # --- Fusion層 (段階的精錬 + カスケード型アフィン変換) ---
        self.fusion_A_to_B = nn.Conv2d(
            hidden_dim_A, hidden_dim_B, kernel_size=3, padding=1
        )
        self.fusion_refine_A = ConvNeXtBlock(
            hidden_dim_B,
            kernel_size=3,
            dilation=1,
            num_groups=8,
            time_dim=self.time_dim,
        )
        self.fusion_mid_refine = ConvNeXtBlock(
            hidden_dim_B,
            kernel_size=3,
            dilation=1,
            num_groups=8,
            time_dim=self.time_dim,
        )
        self.fusion_refine_AB = nn.Sequential(
            nn.GroupNorm(8, hidden_dim_B),
            nn.SiLU(),
            nn.Conv2d(hidden_dim_B, hidden_dim_B, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.fusion_refine_AB_blk = ConvNeXtBlock(
            hidden_dim_B,
            kernel_size=3,
            dilation=1,
            num_groups=8,
            time_dim=self.time_dim,
        )
        self.fusion_AB_to_C_gamma = nn.Conv2d(
            hidden_dim_B, hidden_dim_C, kernel_size=3, padding=1
        )
        self.fusion_AB_to_C_beta = nn.Conv2d(
            hidden_dim_B, hidden_dim_C, kernel_size=3, padding=1
        )

        # --- 最終出力層 (変調後のコントラストを保ったまま等倍エッジを1層精錬) ---
        self.out_norm = nn.GroupNorm(
            num_groups=min(8, hidden_dim_C), num_channels=hidden_dim_C
        )
        self.out_refine = ConvNeXtBlock(
            hidden_dim_C,
            kernel_size=3,
            dilation=1,
            num_groups=8,
            time_dim=self.time_dim,
        )
        self.out_act = nn.GELU()
        self.out_proj = nn.Conv2d(hidden_dim_C, in_channels, kernel_size=3, padding=1)

    def _get_pos_embed(self, embed_dim, h, w, device, freq_scale):
        key = (embed_dim, h, w, str(device), freq_scale)
        pos = self._pos_embed_cache.get(key)
        if pos is None:
            with torch.no_grad():
                pos = get_2d_normalized_sincos_pos_embed(
                    embed_dim, h, w, device, freq_scale=freq_scale
                )
            if len(self._pos_embed_cache) > 32:
                self._pos_embed_cache.clear()
            self._pos_embed_cache[key] = pos
        return pos

    def _pool_stream_a(self, x):
        _, _, H, W = x.shape
        target_h, target_w = max(4, H // 4), max(4, W // 4)
        max_edge = max(target_h, target_w)
        if max_edge > self.max_tokens_A_edge:
            scale = self.max_tokens_A_edge / float(max_edge)
            target_h = max(4, int(round(target_h * scale / 2.0)) * 2)
            target_w = max(4, int(round(target_w * scale / 2.0)) * 2)
        return F.adaptive_avg_pool2d(x, output_size=(target_h, target_w))

    def _pool_stream_b(self, x):
        _, _, H, W = x.shape
        target_h, target_w = max(8, H // 2), max(8, W // 2)
        max_edge_b = self.max_tokens_A_edge * 6
        if max(target_h, target_w) > max_edge_b:
            scale = max_edge_b / float(max(target_h, target_w))
            target_h = max(8, int(round(target_h * scale / 2.0)) * 2)
            target_w = max(8, int(round(target_w * scale / 2.0)) * 2)
        return F.adaptive_avg_pool2d(x, output_size=(target_h, target_w))

    def forward(self, x, context, t):
        B, C, H, W = x.shape

        t_emb = get_timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        t_A = self.t_proj_A(t_emb).unsqueeze(1)  # [B, 1, C_A]
        t_B = self.t_proj_B(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C_B, 1, 1]
        t_C = self.t_proj_C(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C_C, 1, 1]

        # === 1. Split ===
        x_A = self._pool_stream_a(x)
        x_B = self._pool_stream_b(x)
        x_C = x

        pad_mask = context.abs().sum(dim=-1) < 1e-6  # [B, Seq_Len]
        if pad_mask.all(dim=1).any():
            pad_mask = None

        # === 2. Stream A (Global: 2層 Self-Attention) ===
        h_A = self.proj_A(x_A) + t_A.permute(0, 2, 1).unsqueeze(-1)
        _, C_A, H_A, W_A = h_A.shape
        h_A_flat = h_A.view(B, C_A, -1).permute(0, 2, 1)  # [B, H_A*W_A, C_A]

        pos_embed_A = self._get_pos_embed(C_A, H_A, W_A, x.device, freq_scale=64.0)
        h_A_flat = h_A_flat + pos_embed_A.to(h_A_flat.dtype)

        ctx_A = self.text_proj_A(context)
        seq_len_A = h_A_flat.shape[1]
        concat_input = torch.cat([h_A_flat, ctx_A, t_A], dim=1)

        if pad_mask is not None:
            img_mask = torch.zeros((B, seq_len_A), dtype=torch.bool, device=x.device)
            t_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            full_mask_A = torch.cat([img_mask, pad_mask, t_mask], dim=1)
        else:
            full_mask_A = None

        # 1層目 Attention + FFN
        concat_norm = self.norm_A(concat_input)
        attn_out_1, _ = self.attn_A(
            concat_norm, concat_norm, concat_norm, key_padding_mask=full_mask_A
        )
        concat_input = concat_input + attn_out_1
        concat_input = concat_input + self.ffn_A(concat_input)

        # 2層目 Attention + FFN
        concat_norm2 = self.norm_A2(concat_input)
        attn_out_2, _ = self.attn_A2(
            concat_norm2, concat_norm2, concat_norm2, key_padding_mask=full_mask_A
        )
        concat_input = concat_input + attn_out_2
        out_A_flat = concat_input[:, :seq_len_A, :] + self.ffn_A2(
            concat_input[:, :seq_len_A, :]
        )

        out_A = out_A_flat.permute(0, 2, 1).view(B, C_A, H_A, W_A)
        out_A = self.spatial_A(out_A, t_emb)

        # === 3. Stream B (Relational: AdaGN Dilation 1,2,4,8 + テキストFiLM) ===
        if pad_mask is not None:
            valid_mask = (~pad_mask).float().unsqueeze(-1)
            ctx_pooled = (context * valid_mask).sum(dim=1) / valid_mask.sum(
                dim=1
            ).clamp(min=1.0)
        else:
            ctx_pooled = context.mean(dim=1)

        cond_B = self.cond_proj_B(torch.cat([t_emb, ctx_pooled], dim=-1))
        gamma_B, beta_B = cond_B.chunk(2, dim=-1)
        gamma_B = gamma_B.unsqueeze(-1).unsqueeze(-1)
        beta_B = beta_B.unsqueeze(-1).unsqueeze(-1)

        h_B = self.proj_B(x_B) + t_B
        m_B = self.mix_B(h_B)
        for blk in self.blocks_B:
            m_B = blk(m_B, t_emb)
        out_B = h_B + m_B
        out_B = (1.0 + gamma_B) * out_B + beta_B
        out_B = self.post_B(out_B, t_emb)

        # === 4. Pre-Fusion (常に全パラメータ経由の2段階精錬でDDP完全互換) ===
        up_A = F.interpolate(
            out_A,
            size=(out_B.shape[2], out_B.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        up_A = self.fusion_refine_A(self.fusion_A_to_B(up_A), t_emb)
        fusion_AB = up_A + out_B

        mid_h, mid_w = max(fusion_AB.shape[2], H // 2), max(fusion_AB.shape[3], W // 2)
        if fusion_AB.shape[2] != mid_h or fusion_AB.shape[3] != mid_w:
            fusion_AB = F.interpolate(
                fusion_AB, size=(mid_h, mid_w), mode="bilinear", align_corners=False
            )
        fusion_AB = self.fusion_mid_refine(fusion_AB, t_emb)

        up_AB = F.interpolate(
            fusion_AB, size=(H, W), mode="bilinear", align_corners=False
        )
        up_AB = up_AB + self.fusion_refine_AB_blk(self.fusion_refine_AB(up_AB), t_emb)

        # === 5. Stream C (Texture: 高周波位置埋め込み + K/V正規化 Cross-Attention) ===
        h_C = self.proj_C(x_C) + t_C + self.pre_attn_ab_to_c(up_AB)
        _, C_C, H_C, W_C = h_C.shape
        h_C_flat = h_C.view(B, C_C, -1).permute(0, 2, 1)

        pos_embed_C = self._get_pos_embed(C_C, H_C, W_C, x.device, freq_scale=256.0)
        h_C_q = self.norm_C_attn(h_C_flat + pos_embed_C.to(h_C_flat.dtype))

        ctx_proj = self.norm_C_ctx(self.text_proj_C(context))
        need_weights_C = len(self.cross_attn_C._forward_hooks) > 0
        attn_out_C, _ = self.cross_attn_C(
            h_C_q,
            ctx_proj,
            ctx_proj,
            key_padding_mask=pad_mask,
            need_weights=need_weights_C,
        )
        h_C_flat = h_C_flat + attn_out_C
        h_C_flat = h_C_flat + self.ffn_C(h_C_flat)
        h_C_attn = h_C_flat.permute(0, 2, 1).view(B, C_C, H_C, W_C)

        out_C = h_C_attn
        for blk in self.conv_C:
            out_C = blk(out_C, t_emb)

        # === 6. Final Modulation + 等倍高周波エッジ精錬 & Output ===
        gamma = self.fusion_AB_to_C_gamma(up_AB)
        beta = self.fusion_AB_to_C_beta(up_AB)

        normalized_C = self.out_norm(out_C)
        fusion_final = (1.0 + gamma) * normalized_C + beta
        fusion_final = self.out_refine(fusion_final, t_emb)

        output = self.out_proj(self.out_act(fusion_final))
        return output


# 既存コードとの互換エイリアス
TripleStreamPoC = TripleStreamModel
