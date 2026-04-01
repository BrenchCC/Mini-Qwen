import math
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Vision-Module")


@dataclass
class VisionConfig:
    """视觉编码器的配置参数（给小白看的版本）。

    你可以把它理解成：把一张图/一段视频送进视觉编码器之前，需要先把模型结构和切 patch 的规则说清楚。

    关键维度符号（后面注释会反复用到）：
    - B: batch 大小
    - C: 输入通道数（RGB 通常是 3）
    - T: 时间维（视频帧数；纯图片也可以当作 T=1）
    - H/W: 图像高/宽
    - T'/H'/W': 切 patch 后的网格大小
    - D: token 的 embedding 维度（n_embedding）
    """
    n_embedding: int
    n_layer: int
    n_heads: int
    n_output_embed: int
    n_mlp_dim: int
    num_position_embeddings: int

    input_channels: int = 3
    temporal_patch_size: int = 2
    patch_size: int = 16
    spatial_merge_size: int = 2

class VisionRotaryEmbedding(nn.Module):
    """Vision Rotary Embedding"""
    def __init__(
        self,
        dim: int,
        theta: float = 10000.0
    ) -> None:
        super().__init__()
        # dim：这里表示要生成 RoPE 频率表的“半维度”
        # 例如 attention 的 head_dim=128 时：
        # - 2D RoPE 通常先按 h/w 两个轴分别生成 head_dim/4 的频率
        # - 拼起来变成 head_dim/2（对应 frequency 的最后一维）
        #
        # theta：RoPE 的基底常数，越大表示不同维度的频率跨度越大（经典值 10000）
        inverse_frequency = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.float) / dim))
        self.register_buffer("inverse_frequency", inverse_frequency, persistent = False)

    def forward(self, seq_len: int) -> torch.Tensor:    
        """生成 RoPE 的角频率表（frequency table）。

        说明（给小白看的）：
        - RoPE 不是直接存「位置 embedding 向量」，而是为每个位置生成一组旋转角度。
        - 这里返回的是 `frequency`，后面会在 attention 里做 `cos/sin` 并旋转 q/k。

        维度：
        - 输入：`seq_len`（序列长度 L）
        - 输出：`frequency` 形状为 `(L, dim/2)`
          其中 `dim` 是 head_dim 或 head_dim/2（取决于你怎么设计 RoPE）。
        """
        sequence = torch.arange(seq_len, dtype = self.inverse_frequency.dtype, device = self.inverse_frequency.device)
        frequency = sequence[:, None] * self.inverse_frequency[None, :]

        return frequency
    

class VisionPatchEmbedding(nn.Module):
    """Vision Patch Embedding"""
    def __init__(
        self,
        config: VisionConfig
    ) -> None:
        super().__init__()
        self.config = config
        # 3D 卷积的 kernel/stride：
        # - temporal_patch_size：沿时间维一次吃多少帧（类似视频 patch）
        # - patch_size：沿空间 H/W 一次吃多少像素（类似 ViT 的 patch）
        self.kernel_size = [self.config.temporal_patch_size, self.config.patch_size, self.config.patch_size]
        self.stride = [self.config.temporal_patch_size, self.config.patch_size, self.config.patch_size]
       
        self.projection = nn.Conv3d(
            in_channels = self.config.input_channels,
            out_channels = self.config.n_embedding,
            kernel_size = self.kernel_size,
            stride = self.stride,
            bias = True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project pixels to patch embeddings.

        Parameters
        ----------
        x: torch.Tensor
            Pixel tensor in shape (batch, channels, time, height, width).

        Returns
        -------
        torch.Tensor
            Packed patch embeddings in shape (total_patches, dim). The patch order is
            re-arranged to make each `spatial_merge_size x spatial_merge_size` group
            contiguous, matching the downstream `PatchMerger` expectation.
        """
        # 期望输入是 5D：
        # x: (B, C, T, H, W)
        # B=batch，C=通道数，T=时间维（视频/帧数），H/W=图像高宽
        if x.ndim != 5:
            raise ValueError(f"Expected pixels in (B, C, T, H, W), got shape = {tuple(x.shape)}")

        # 1) 用 Conv3d 把像素切成 patch，并投影到 embedding 维度
        # projection 输出： (B, D, T', H', W')
        # 其中：
        # - D = n_embedding
        # - T' = T / temporal_patch_size
        # - H' = H / patch_size
        # - W' = W / patch_size
        x = self.projection(x)
        # 2) 把通道维挪到最后，方便后续 reshape/merge
        # (B, D, T', H', W') -> (B, T', H', W', D)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        batch_size, t_grid, h_grid, w_grid, dim = x.shape
        merge_size = int(self.config.spatial_merge_size)
        if (h_grid % merge_size) != 0 or (w_grid % merge_size) != 0:
            raise ValueError(
                f"Patch grid (H'={h_grid}, W'={w_grid}) must be divisible by spatial_merge_size={merge_size}"
            )

        # 3) 为了让后面的 PatchMerger 能一次性合并 (merge_size x merge_size) 的空间 patch，
        #    这里把 H'/W' 拆成：外层网格 + 内层 merge 小块
        # (B, T', H', W', D)
        # -> (B, T', H'/m, m, W'/m, m, D)
        x = x.view(
            batch_size,
            t_grid,
            h_grid // merge_size,
            merge_size,
            w_grid // merge_size,
            merge_size,
            dim,
        )
        # 4) 调整维度顺序，让每个 (m x m) 的小块在内存里是连续的
        # (B, T', H'/m, m, W'/m, m, D)
        # -> (B, T', H'/m, W'/m, m, m, D)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)

        # 5) 最终把所有 token “打包”成一条长序列
        # token 数量 = B * T' * H' * W'
        # (B, T', H'/m, W'/m, m, m, D) -> (B*T'*H'*W', D)
        x = x.contiguous().view(batch_size * t_grid * h_grid * w_grid, dim)
        return x

class PatchMerger(nn.Module):
    """Patch Merger"""
    def __init__(
        self,
        config: VisionConfig,
        use_post_shuffle_norm: bool = False
    ) -> None:
        super().__init__()
        self.config = config

        # Merge spatial_merge_size x spatial_merge_size patches in spatial dimensions.
        # hidden_size = D * (m*m)
        # 因为我们要把一个 m×m 的空间小块里的 patch token 拼接到一起当作一个更大的 token。
        self.hidden_size = config.n_embedding * (config.spatial_merge_size ** 2)
        self.use_post_shuffle_norm = use_post_shuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_post_shuffle_norm else config.n_embedding, 
            eps = 1e-6,
        )
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(self.hidden_size, self.config.n_output_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把多个空间 patch 合并成一个 token，并投影到输出维度。

        这里的输入 x 通常是把一个 (m x m) 的空间小块 flatten 后的向量。

        维度：
        - 输入：`x` 形状通常是 `(num_groups, m*m*D)`
          - `num_groups` = B * T' * (H'/m) * (W'/m)
          - `m` = spatial_merge_size
          - `D` = n_embedding
        - 输出：形状 `(num_groups, n_output_embed)`
        """
        x = self.norm(
            x.contiguous().view(-1, self.hidden_size) if self.use_post_shuffle_norm else x
        ).view(-1, self.hidden_size)
        # (num_groups, m*m*D) -> (num_groups, hidden_size)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        
        return x

class VisionAttention(nn.Module):    
    def __init__(
        self,
        config: VisionConfig    
    ) -> None:
        super().__init__()
        # 线性层一次性生成 q/k/v（省一次 matmul）
        self.qkv = nn.Linear(config.n_embedding, 3 * config.n_embedding, bias = True)
        self.n_heads = config.n_heads
        self.head_dim = config.n_embedding // config.n_heads
        # Must project back to n_embedding for residual add.
        self.projection = nn.Linear(config.n_embedding, config.n_embedding, bias = True)

    @staticmethod
    def _rotary_half(x: torch.Tensor) -> torch.Tensor:
        """Rotary Half"""
        # 把最后一维拆成两半 (x1, x2)，并返回 (-x2, x1)。
        # 这是实现 2D 旋转的一个常用小技巧：rotate_half(x)。
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]

        return torch.cat((-x2, x1), dim = -1)   
    
    @staticmethod
    def _apply_rotary_position_embedding_vision(x: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        """对 q/k 应用 RoPE（旋转位置编码）。

        维度（关键点）：
        - x: `(L, n_heads, head_dim)`
        - frequency: `(L, head_dim/2)`
        - 输出：与 x 同形状 `(L, n_heads, head_dim)`

        小白理解版：
        - 把向量拆成两半 (x1, x2)
        - 通过 cos/sin 做 2D 旋转：x*cos + rotate_half(x)*sin
        """
        if frequency is None:
            return x

        # x: (seq_len, n_heads, head_dim)
        # frequency: (seq_len, head_dim / 2)
        original_type = x.dtype
        x = x.float()

        cos = torch.cos(frequency).float()
        sin = torch.sin(frequency).float()

        cos = torch.cat([cos, cos], dim = -1).unsqueeze(1)  # (seq_len, 1, head_dim)
        sin = torch.cat([sin, sin], dim = -1).unsqueeze(1)  # (seq_len, 1, head_dim)

        output = (x * cos) + (VisionAttention._rotary_half(x) * sin)
        return output.to(original_type)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor = None, # A special index tensor (cu_seqlens) is used to keep track of the boundary positions of the individual original sequences
        rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        """自注意力（仅做同一张图/同一段序列内部的 attention）。

        输入输出维度：
        - 输入 x: `(L, D)`，L 是打包后的 token 总数，D=n_embedding
        - 输出: `(L, D)`，方便 residual 相加

        注意：
        - 这里用 `cu_seqlens` 生成 block-diagonal mask，让不同样本/不同帧之间互不 attention。
        """
        seq_len = x.shape[0]
        # 线性层一次性算出 q/k/v
        # (L, D) -> (L, 3*D) -> (L, 3, n_heads, head_dim)
        q, k, v = (
            self.qkv(x)
            .reshape(seq_len, 3, self.n_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        # 对 q/k 应用 RoPE（v 一般不需要旋转）
        q = self._apply_rotary_position_embedding_vision(q, rotary_pos_emb)
        k = self._apply_rotary_position_embedding_vision(k, rotary_pos_emb)

        # (seq, heads, dim) -> (heads, seq, dim)
        # 下面这样做的原因：matmul 计算 attention 时通常按 (H, L, d) 来写更直观。
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # 注意力分数：QK^T / sqrt(d)
        # (H, L, d) x (H, d, L) -> (H, L, L)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if cu_seqlens is not None:
            # 生成 block-diagonal mask：
            # 不同子序列之间填 -inf，softmax 后就等于 0
            # cu_seqlens 是“前缀和”的边界列表：
            # - 例如有 3 个子序列，长度分别为 [5, 7, 2]
            # - 那 cu_seqlens = [0, 5, 12, 14]
            attention_mask = torch.full(
                [1, seq_len, seq_len],
                torch.finfo(attn_weights.dtype).min,
                device = attn_weights.device,
                dtype = attn_weights.dtype,
            )
            for i in range(1, int(cu_seqlens.numel())):
                start = int(cu_seqlens[i - 1].item())
                end = int(cu_seqlens[i].item())
                attention_mask[..., start:end, start:end] = 0
            attn_weights = attn_weights + attention_mask

        # softmax 归一化
        attn_weights = F.softmax(attn_weights, dim = -1, dtype = torch.float32).to(q.dtype)
        # 加权求和： (H, L, L) x (H, L, d) -> (H, L, d)
        attn_output = torch.matmul(attn_weights, v)

        # (heads, seq, dim) -> (seq, heads, dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, -1)
        # (L, H*d) -> (L, D)
        return self.projection(attn_output)

class VisionMLP(nn.Module):
    """Vision MLP"""
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embedding, config.n_mlp_dim, bias = True)
        self.activation = nn.GELU(approximate = "tanh")
        self.linear_2 = nn.Linear(config.n_mlp_dim, config.n_embedding, bias = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """两层前馈网络（MLP）。

        维度：
        - 输入 x: `(L, D)`
        - 中间: `(L, n_mlp_dim)`
        - 输出: `(L, D)`
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x

class VisionBlock(nn.Module):
    """Vision Block"""
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.attention = VisionAttention(config)
        self.mlp = VisionMLP(config)

        self.norm_1 = nn.LayerNorm(config.n_embedding, eps = 1e-6)
        self.norm_2 = nn.LayerNorm(config.n_embedding, eps = 1e-6)

    def forward(self, x, cu_seqlens = None, rotary_pos_emb = None) -> torch.Tensor:
        """一个标准 Transformer block（Attention + MLP）。

        维度：
        - 输入/输出 x: `(L, D)`
        """
        x = x + self.attention(
            self.norm_1(x), 
            cu_seqlens = cu_seqlens,
            rotary_pos_emb = rotary_pos_emb,
        )
        x = x + self.mlp(self.norm_2(x))
        return x

    


class VisionEncoder(nn.Module):
    """Vision Encoder"""
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = VisionPatchEmbedding(config)
        # Reference-compatible alias
        self.patch_embed = self.patch_embedding

        # Learnable position embeddings
        # 这里的 position_embeddings 是“基础网格”的可学习参数：
        # - 形状：(num_position_embeddings, D)
        # - 通常假设它对应一个 sqrt(N) × sqrt(N) 的二维网格
        # - 输入图片/视频的 patch 网格尺寸不一定等于这个基础网格，所以需要插值
        self.position_embeddings = nn.Embedding(config.num_position_embeddings, config.n_embedding)
        # Reference-compatible alias
        self.pos_embed = self.position_embeddings
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        if (self.num_grid_per_side * self.num_grid_per_side) != int(config.num_position_embeddings):
            raise ValueError(
                "num_position_embeddings must be a perfect square for 2D grid interpolation, "
                f"got num_position_embeddings={config.num_position_embeddings}"
            )

        self.blocks = nn.ModuleList(
            [VisionBlock(config) for _ in range(config.n_layer)]
        )

        self.merger = PatchMerger(config = config, use_post_shuffle_norm = True)

        self.head_dim = config.n_embedding // config.n_heads
        if (self.head_dim * config.n_heads) != config.n_embedding:
            raise ValueError(
                "n_embedding must be divisible by n_heads, "
                f"got n_embedding={config.n_embedding}, n_heads={config.n_heads}"
            )
        if (self.head_dim % 2) != 0:
            raise ValueError(
                "head_dim must be even for rotary embedding, "
                f"got head_dim={self.head_dim}"
            )

        # For 2D rotary: we generate per-axis frequencies of size head_dim/4, then
        # concatenate (h, w) => head_dim/2, which matches `_apply_rotary_position_embedding_vision`.
        self.rotary_position_embedding = VisionRotaryEmbedding(dim = self.head_dim // 2)
        # Reference-compatible alias
        self.rotary_pos_emb = self.rotary_position_embedding

        self.spatial_merge_size = config.spatial_merge_size

    def fast_pos_embed_interpolate(self, d_image: torch.Tensor) -> torch.Tensor:
        """Interpolate learned position embeddings to match patch-grid dimensions.

        Parameters
        ----------
        d_image: torch.Tensor
            Patch-grid shape tensor in (batch, 3): (grid_t, grid_h, grid_w).

        Returns
        -------
        torch.Tensor
            Packed position embeddings in shape (total_patches, dim). The ordering is
            consistent with `VisionPatchEmbedding.forward` (merge-group contiguous).
        """
        grid_ts, grid_hs, grid_ws = d_image[:, 0], d_image[:, 1], d_image[:, 2]
        device = d_image.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        # 目标：对每个样本把 (num_position_embeddings) 这张“基础网格”插值到当前 (h, w) 的 patch 网格。
        # 然后再重复 t 次，并且按照 merge 的顺序 permute 成“打包 token”的顺序。
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h = int(h.item())
            w = int(w.item())

            # 生成插值坐标：把目标网格 (h, w) 均匀映射到基础网格 [0, base-1]
            # h_idxs/w_idxs: (h,) / (w,)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device = device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device = device)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max = self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max = self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            # 双线性插值：每个目标点会从基础网格的 4 个邻居取值
            # indices[i]: (h*w,)
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            # 对应 4 个邻居的权重，weights[i]: (h*w,)
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype = torch.long, device = device)
        weight_tensor = torch.tensor(weight_list, dtype = self.pos_embed.weight.dtype, device = device)

        # pos_embed(idx_tensor): (4, total_hw, D)
        # weight_tensor[:, :, None]: (4, total_hw, 1)
        # 广播相乘后还是 (4, total_hw, D)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        # 4 个邻居加权求和 -> (total_hw, D)
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # split 回每个样本各自的 (h*w, D)，方便后面 repeat(t, 1) 和按 merge 顺序重排
        patch_pos_embeds = patch_pos_embeds.split([int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = int(self.config.spatial_merge_size)
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t = int(t.item())
            h = int(h.item())
            w = int(w.item())

            # 先把 2D 的位置 embedding (h*w, D) 重复 t 次 -> (t*h*w, D)
            pos_embed = pos_embed.repeat(t, 1)
            # 按 merge_size 重排，让 (m x m) 小块连续：
            # (t*h*w, D)
            # -> view(t, h/m, m, w/m, m, D)
            # -> permute(t, h/m, w/m, m, m, D)
            # -> flatten(0, 4) => (t*h*w, D)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return torch.cat(patch_pos_embeds_permute, dim = 0)

    # Backward-compatible alias
    def fast_position_embedding_interpolate(self, dim_image: torch.Tensor) -> torch.Tensor:
        return self.fast_pos_embed_interpolate(dim_image)
    
    def rot_pos_emb(self, d_image: torch.Tensor) -> torch.Tensor:
        """Build 2D rotary position embeddings for packed vision tokens.

        Parameters
        ----------
        d_image: torch.Tensor
            Patch-grid shape tensor in (batch, 3): (grid_t, grid_h, grid_w).

        Returns
        -------
        torch.Tensor
            Rotary frequency tensor in shape (total_patches, head_dim / 2).
        """
        pos_ids = []
        sms = int(self.spatial_merge_size)

        # 目标：为每个 token 生成 2D 位置（h_id, w_id），再查表得到对应的 RoPE 频率。
        # 注意这里的位置顺序同样要匹配 merge 后的 token 顺序。
        for t, h, w in d_image:
            t = int(t.item())
            h = int(h.item())
            w = int(w.item())

            # hpos_ids/wpos_ids 初始是标准的二维网格坐标
            # hpos_ids: (h, w) 表示每个位置的行号
            hpos_ids = torch.arange(h, device = d_image.device).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2)
            hpos_ids = hpos_ids.flatten()

            # wpos_ids: (h, w) 表示每个位置的列号
            wpos_ids = torch.arange(w, device = d_image.device).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2)
            wpos_ids = wpos_ids.flatten()

            # (h*w, 2) -> repeat(t, 1) => (t*h*w, 2)
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim = -1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim = 0)
        max_grid_size = int(d_image[:, 1:].max().item())

        # rotary_pos_emb_full: (max_grid_size, head_dim/4)
        # 通过 pos_ids 取出每个 token 的 (h, w) 对应频率：
        # rotary_pos_emb_full[pos_ids]: (total_tokens, 2, head_dim/4)
        # flatten(1) -> (total_tokens, head_dim/2)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    # Backward-compatible alias
    def rotate_position_embedding(self, dim_image: torch.Tensor) -> torch.Tensor:
        return self.rotary_position_embedding(int(dim_image))

    
    def forward(
        self,
        pixels: torch.Tensor,
        d_image: torch.Tensor = None,
        dim_image: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode pixels into merged vision embeddings (reference-compatible).

        Parameters
        ----------
        pixels: torch.Tensor
            Pixel tensor in shape (batch, channels, time, height, width).
        d_image: torch.Tensor
            Patch-grid shape tensor in (batch, 3): (grid_t, grid_h, grid_w).
        dim_image: torch.Tensor
            Deprecated alias for `d_image`.

        Returns
        -------
        torch.Tensor
            Merged vision embeddings in shape (num_groups, output_dim).
        """
        # d_image 形状：(B, 3)
        # 每行：(grid_t, grid_h, grid_w)
        # - grid_t = T'
        # - grid_h = H'
        # - grid_w = W'
        if d_image is None:
            d_image = dim_image
        if d_image is None:
            # Auto-infer for fixed-size batches.
            # 注意：只有当一个 batch 内所有样本的 (T, H, W) 相同时，这个推断才是安全的。
            batch_size, _, t, h, w = pixels.shape
            d_image = torch.tensor(
                [[t // self.config.temporal_patch_size, h // self.config.patch_size, w // self.config.patch_size]]
                * batch_size,
                device = pixels.device,
                dtype = torch.long,
            )

        # 1) patch_embed：把像素变成 token，并且按 merge 顺序打包
        # pixels: (B, C, T, H, W)
        # hidden_states: (total_tokens, D)
        # total_tokens = sum_i (grid_t * grid_h * grid_w)
        hidden_states = self.patch_embed(pixels)

        expected_tokens = int((d_image[:, 0] * d_image[:, 1] * d_image[:, 2]).sum().item())
        if hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                "Packed token count mismatch: "
                f"got hidden_states.shape[0]={hidden_states.shape[0]}, expected={expected_tokens}. "
                "Check that `d_image` matches the patch grid of `pixels`."
            )

        # 2) 加可学习的位置 embedding（插值到当前 patch 网格）
        # pos_embeds: (total_tokens, D)
        pos_embeds = self.fast_pos_embed_interpolate(d_image).to(hidden_states.dtype)
        hidden_states = hidden_states + pos_embeds

        # 3) 构造 2D RoPE（给 attention 用）
        # rotary_pos_emb: (total_tokens, head_dim/2)
        rotary_pos_emb = self.rot_pos_emb(d_image)

        # 4) 构造 cu_seqlens（每个“子序列”的边界，用来做 block-diagonal attention mask）
        # 这里的“子序列”粒度是：每个样本的每一帧（t 次），每帧长度是 grid_h*grid_w
        # repeat_interleave(d_image[:,1]*d_image[:,2], d_image[:,0])
        # -> (sum(grid_t),) 的每帧长度列表
        # cumsum + pad => (sum(grid_t)+1,)
        cu_seqlens = torch.repeat_interleave(d_image[:, 1] * d_image[:, 2], d_image[:, 0]).cumsum(
            dim = 0,
            dtype = torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value = 0)

        # 5) Transformer blocks：输入输出都保持 (total_tokens, D)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens = cu_seqlens, rotary_pos_emb = rotary_pos_emb)

        # 6) PatchMerger：把每个 (m*m) 空间小块合并成一个 token，并投影到 n_output_embed
        # hidden_states: (total_tokens, D)
        # 输出： (num_groups, n_output_embed)
        # num_groups = total_tokens / (m*m)
        return self.merger(hidden_states)