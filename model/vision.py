import math
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Vision-Module")


@dataclass
class VisionConfig:
    """视觉编码器配置。

    主要描述两件事：
    1) 模型结构：Transformer 的层数/头数/隐藏维度等。
    2) Patch 规则：时间维与空间维如何切 patch，以及后续如何做空间合并。

    维度符号约定（本文件注释统一使用）：
    - B: batch 大小
    - C: 输入通道数（RGB 通常为 3）
    - T: 时间维（视频帧数；静态图片可视作 T=1）
    - H/W: 输入图像高/宽
    - T'/H'/W': 经过 patch 切分后的网格尺寸
    - D: token embedding 维度（即 `n_embedding`）
    """

    n_embedding: int  # D，视觉 token 的 embedding 维度
    n_layer: int  # Transformer block 层数
    n_heads: int  # Attention 头数（需整除 n_embedding）
    n_output_embed: int  # PatchMerger 输出维度（供下游语言模型/融合模块使用）
    n_mlp_dim: int  # MLP 中间层维度
    num_position_embeddings: int  # 可学习 2D 位置表大小（要求为完全平方数）

    input_channels: int = 3  # 输入像素通道数（RGB=3）
    temporal_patch_size: int = 2  # 时间维 patch 大小：一次合并多少帧
    patch_size: int = 16  # 空间维 patch 大小：一次合并多少像素（H/W）
    spatial_merge_size: int = 2  # 空间合并因子 m：后续将 (m*m) 个 patch 合并为 1 个 token

class VisionRotaryEmbedding(nn.Module):
    """视觉侧 RoPE（Rotary Position Embedding）频率表生成器。

    这里不直接生成“位置向量”，而是生成每个位置对应的角频率表（frequency）。
    后续在 attention 中通过 `cos/sin` 将 q/k 做旋转，从而注入位置信息。
    """
    def __init__(
        self,
        dim: int,
        theta: float = 10000.0
    ) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.float) / dim))  # RoPE 逆频率表（长度约为 dim/2）
        # 将逆频率表缓存为 buffer（不参与训练，随 device 迁移）
        self.register_buffer("inverse_frequency", inverse_frequency, persistent = False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """生成 RoPE 角频率表（frequency）。

        参数
        - seq_len: 序列长度 L。

        返回
        - frequency: `(L, dim/2)`，供后续计算 `cos/sin` 并对 q/k 做旋转。

        备注
        - 这里的 `dim` 通常对应 `head_dim/2`（2D RoPE 会在 h/w 两个轴各生成一段频率后拼接）。
        """
        sequence = torch.arange(seq_len, dtype = self.inverse_frequency.dtype, device = self.inverse_frequency.device)  # (L,)
        frequency = sequence[:, None] * self.inverse_frequency[None, :]  # (L, dim/2)

        return frequency
    

class VisionPatchEmbedding(nn.Module):
    """将像素切 patch 并投影为视觉 token。

    采用 `Conv3d` 同时处理时间维与空间维：
    - 时间维 kernel/stride 为 `temporal_patch_size`
    - 空间维 kernel/stride 为 `patch_size`
    """
    def __init__(
        self,
        config: VisionConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.kernel_size = [self.config.temporal_patch_size, self.config.patch_size, self.config.patch_size]  # Conv3d kernel: (T_patch, H_patch, W_patch)
        self.stride = [self.config.temporal_patch_size, self.config.patch_size, self.config.patch_size]  # Conv3d stride: 与 kernel 相同以实现不重叠切 patch
       
        self.projection = nn.Conv3d(
            in_channels = self.config.input_channels,
            out_channels = self.config.n_embedding,
            kernel_size = self.kernel_size,
            stride = self.stride,
            bias = True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把像素张量转换为 patch token，并按 merge 规则重排。

        参数
        - x: `(B, C, T, H, W)` 的像素张量。

        返回
        - packed_tokens: `(total_tokens, D)` 的 token 序列，其中 token 顺序已重排，保证每个
          `spatial_merge_size x spatial_merge_size` 空间小块对应的 patch 在内存中连续，便于下游 `PatchMerger` 合并。
        """
        # 关键约束：视觉输入必须是 5D `(B, C, T, H, W)`
        if x.ndim != 5:
            raise ValueError(f"Expected pixels in (B, C, T, H, W), got shape = {tuple(x.shape)}")

        # 1) Conv3d：切 patch + 投影到 embedding 维度
        # 输出形状为 `(B, D, T', H', W')`
        x = self.projection(x)
        # 2) 调整维度顺序：把 embedding 维移到最后，便于后续 reshape/merge
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        batch_size, t_grid, h_grid, w_grid, dim = x.shape  # (B, T', H', W', D)
        merge_size = int(self.config.spatial_merge_size)  # m：每个空间小块边长（m*m 个 patch 合并）
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
    """空间 patch 合并器。

    将 `spatial_merge_size x spatial_merge_size` 的空间小块内的 patch token 拼接起来，
    通过 MLP 投影为一个更粗粒度的 token，降低序列长度。
    """
    def __init__(
        self,
        config: VisionConfig,
        use_post_shuffle_norm: bool = False
    ) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = config.n_embedding * (config.spatial_merge_size ** 2)  # hidden_size = D * (m*m)，拼接一个空间小块内的 patch token
        self.use_post_shuffle_norm = use_post_shuffle_norm
        # 可选：在合并后的向量上做 LayerNorm（常用于 token shuffle/重排之后稳定分布）
        self.norm = nn.LayerNorm(
            self.hidden_size if use_post_shuffle_norm else config.n_embedding, 
            eps = 1e-6,
        )
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(self.hidden_size, self.config.n_output_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把多个空间 patch 合并成一个 token，并投影到输出维度。

        参数
        - x: 通常为 `(num_groups, m*m*D)`。

        返回
        - out: `(num_groups, n_output_embed)`。

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
    """视觉侧自注意力（支持 2D RoPE 与 block-diagonal mask）。"""
    def __init__(
        self,
        config: VisionConfig    
    ) -> None:
        super().__init__()
        self.qkv = nn.Linear(config.n_embedding, 3 * config.n_embedding, bias = True)  # 一次性生成 q/k/v，减少一次线性层调用
        self.n_heads = config.n_heads  # 注意力头数 H
        self.head_dim = config.n_embedding // config.n_heads  # 每个头的维度 d
        # residual 需要回到 D 维，以便与输入相加
        self.projection = nn.Linear(config.n_embedding, config.n_embedding, bias = True)

    @staticmethod
    def _rotary_half(x: torch.Tensor) -> torch.Tensor:
        """RoPE 辅助函数：对最后一维做 (x1, x2) -> (-x2, x1) 变换。"""
        half = x.shape[-1] // 2  # head_dim 的一半
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
        original_type = x.dtype  # 计算 cos/sin 时先转 float32，最后再转回原 dtype
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
        cu_seqlens: torch.Tensor = None,
        rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        """自注意力（仅做同一张图/同一段序列内部的 attention）。

        输入输出维度：
        - 输入 x: `(L, D)`，L 是打包后的 token 总数，D=n_embedding
        - 输出: `(L, D)`，方便 residual 相加

        参数
        - x: `(L, D)` 打包后的 token 序列。
        - cu_seqlens: 可选的边界索引（前缀和形式），用于构造 block-diagonal mask，
          让不同样本/不同帧之间互不 attention。
        - rotary_pos_emb: 可选的 RoPE 频率表 `(L, head_dim/2)`，用于旋转 q/k。
        """
        seq_len = x.shape[0]  # L，总 token 数
        # 线性层一次性算出 q/k/v，并 reshape 为按头拆分的形状
        q, k, v = (
            self.qkv(x)
            .reshape(seq_len, 3, self.n_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        # 关键操作：对 q/k 应用 RoPE（v 一般不需要旋转）
        q = self._apply_rotary_position_embedding_vision(q, rotary_pos_emb)
        k = self._apply_rotary_position_embedding_vision(k, rotary_pos_emb)

        # 关键操作：转置到 `(H, L, d)`，便于后续计算 QK^T 与加权求和
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
    """视觉侧两层前馈网络（MLP）。"""
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embedding, config.n_mlp_dim, bias = True)
        self.activation = nn.GELU(approximate = "tanh")
        self.linear_2 = nn.Linear(config.n_mlp_dim, config.n_embedding, bias = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """两层前馈网络（MLP）。

        参数
        - x: `(L, D)` 的 token 序列。

        返回
        - out: `(L, D)`。

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
    """视觉侧 Transformer Block（Pre-Norm Attention + Pre-Norm MLP）。"""
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.attention = VisionAttention(config)
        self.mlp = VisionMLP(config)

        self.norm_1 = nn.LayerNorm(config.n_embedding, eps = 1e-6)
        self.norm_2 = nn.LayerNorm(config.n_embedding, eps = 1e-6)

    def forward(self, x, cu_seqlens = None, rotary_pos_emb = None) -> torch.Tensor:
        """一个标准 Transformer block（Attention + MLP）。

        参数
        - x: `(L, D)` 的 token 序列。
        - cu_seqlens: 可选边界索引（前缀和），用于 attention 的 block-diagonal mask。
        - rotary_pos_emb: 可选 RoPE 频率表 `(L, head_dim/2)`。

        返回
        - out: `(L, D)`。
        """
        x = x + self.attention(
            self.norm_1(x), 
            cu_seqlens = cu_seqlens,
            rotary_pos_emb = rotary_pos_emb,
        )
        x = x + self.mlp(self.norm_2(x))
        return x

    


class VisionEncoder(nn.Module):
    """视觉编码器：patch embedding + position embedding + Transformer + patch merge。"""
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = VisionPatchEmbedding(config)  # 像素 -> patch token（并按 merge 规则重排）
        # Reference-compatible alias
        self.patch_embed = self.patch_embedding

        self.position_embeddings = nn.Embedding(config.num_position_embeddings, config.n_embedding)  # 可学习 2D 基础网格（后续会插值到当前 patch 网格）
        # Reference-compatible alias
        self.pos_embed = self.position_embeddings
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)  # 基础网格边长 sqrt(num_position_embeddings)
        if (self.num_grid_per_side * self.num_grid_per_side) != int(config.num_position_embeddings):
            raise ValueError(
                "num_position_embeddings must be a perfect square for 2D grid interpolation, "
                f"got num_position_embeddings={config.num_position_embeddings}"
            )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [VisionBlock(config) for _ in range(config.n_layer)]
        )

        self.merger = PatchMerger(config = config, use_post_shuffle_norm = True)  # (m*m) patch token -> 1 个 merged token

        self.head_dim = config.n_embedding // config.n_heads  # 每个头的维度 d
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

        self.rotary_position_embedding = VisionRotaryEmbedding(dim = self.head_dim // 2)  # 2D RoPE：每个轴生成 head_dim/4，拼接后为 head_dim/2
        # Reference-compatible alias
        self.rotary_pos_emb = self.rotary_position_embedding

        self.spatial_merge_size = config.spatial_merge_size  # m：空间合并因子

    def fast_pos_embed_interpolate(self, d_image: torch.Tensor) -> torch.Tensor:
        """将可学习的 2D 位置表插值到当前 patch 网格，并按 packed token 顺序输出。

        参数
        - d_image: `(B, 3)`，每行是 `(grid_t, grid_h, grid_w)`。

        返回
        - pos_embeds: `(total_tokens, D)`，顺序与 `VisionPatchEmbedding.forward` 的 packed token 顺序一致。
        """
        grid_ts, grid_hs, grid_ws = d_image[:, 0], d_image[:, 1], d_image[:, 2]  # (B,) 每个样本的 (T', H', W')
        device = d_image.device

        idx_list = [[] for _ in range(4)]  # 4 个邻居位置索引（双线性插值）
        weight_list = [[] for _ in range(4)]  # 4 个邻居对应权重

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

        idx_tensor = torch.tensor(idx_list, dtype = torch.long, device = device)  # (4, total_hw)
        weight_tensor = torch.tensor(weight_list, dtype = self.pos_embed.weight.dtype, device = device)  # (4, total_hw)

        # pos_embed(idx_tensor): (4, total_hw, D)
        # weight_tensor[:, :, None]: (4, total_hw, 1)
        # 广播相乘后还是 (4, total_hw, D)
        # 关键操作：按索引取 embedding，并乘上插值权重
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        # 4 个邻居加权求和 -> (total_hw, D)
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # split 回每个样本各自的 (h*w, D)，方便后面 repeat(t, 1) 和按 merge 顺序重排
        patch_pos_embeds = patch_pos_embeds.split([int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = int(self.config.spatial_merge_size)  # m：空间合并因子
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
        """兼容旧接口：等价于 `fast_pos_embed_interpolate`。"""
        return self.fast_pos_embed_interpolate(dim_image)
    
    def rot_pos_emb(self, d_image: torch.Tensor) -> torch.Tensor:
        """为 packed token 构造 2D RoPE 频率表。

        参数
        - d_image: `(B, 3)`，每行是 `(grid_t, grid_h, grid_w)`。

        返回
        - rotary_pos_emb: `(total_tokens, head_dim/2)`，用于 attention 中旋转 q/k。
        """
        pos_ids = []
        sms = int(self.spatial_merge_size)  # m：空间合并因子

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
        """兼容旧接口：保留原实现（注意该接口不等价于 2D RoPE 构造）。"""
        return self.rotary_position_embedding(int(dim_image))

    
    def forward(
        self,
        pixels: torch.Tensor,
        d_image: torch.Tensor = None,
        dim_image: torch.Tensor = None,
    ) -> torch.Tensor:
        """将像素编码为 merged vision embeddings。

        参数
        - pixels: `(B, C, T, H, W)` 像素张量。
        - d_image: 可选 `(B, 3)` patch 网格信息，每行是 `(grid_t, grid_h, grid_w)`。
        - dim_image: 兼容旧参数名，等价于 `d_image`。

        返回
        - merged_embeds: `(num_groups, n_output_embed)`，其中 `num_groups = total_tokens / (m*m)`。
        """
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

        expected_tokens = int((d_image[:, 0] * d_image[:, 1] * d_image[:, 2]).sum().item())  # total_tokens = sum_i(T'_i*H'_i*W'_i)
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


def _smoke_test() -> None:
    """最小可运行的 smoke test。

    目标：
    - 覆盖 `VisionEncoder.forward` 的两种用法：显式传 `d_image` / 让模型自动推断。
    - 校验输出形状与关键中间张量形状。
    """
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU（如果可用），否则使用 CPU
    dtype = torch.float32  # 测试用 dtype：保持 float32 便于排查数值问题

    config = VisionConfig(
        n_embedding = 32,
        n_layer = 2,
        n_heads = 4,
        n_output_embed = 64,
        n_mlp_dim = 64,
        num_position_embeddings = 16,
        input_channels = 3,
        temporal_patch_size = 2,
        patch_size = 16,
        spatial_merge_size = 2,
    )

    encoder = VisionEncoder(config).to(device)  # 实例化视觉编码器
    encoder.eval()

    batch_size = 2  # B：batch 大小
    channels = config.input_channels  # C：输入通道数
    t = 2  # T：帧数（时间维）；需能被 temporal_patch_size 整除
    h = 32  # H：输入图像高度；需能被 patch_size 整除
    w = 32  # W：输入图像宽度；需能被 patch_size 整除

    pixels = torch.randn(batch_size, channels, t, h, w, device = device, dtype = dtype)  # (B, C, T, H, W)

    # 关键操作：显式提供 patch 网格信息
    # - T' = T / temporal_patch_size
    # - H' = H / patch_size
    # - W' = W / patch_size
    d_image = torch.tensor(
        [[t // config.temporal_patch_size, h // config.patch_size, w // config.patch_size]] * batch_size,
        device = device,
        dtype = torch.long,
    )

    with torch.no_grad():
        # 关键操作：分别测试显式传参与自动推断，二者结果形状应一致
        out_explicit = encoder(pixels, d_image = d_image)
        out_infer = encoder(pixels, d_image = None)

        # 关键形状检查：用 d_image 计算 packed token 总数
        expected_tokens = int((d_image[:, 0] * d_image[:, 1] * d_image[:, 2]).sum().item())  # sum_i(T'_i*H'_i*W'_i)
        pos = encoder.fast_pos_embed_interpolate(d_image)  # (total_tokens, D)
        rope = encoder.rot_pos_emb(d_image)  # (total_tokens, head_dim/2)

    # 断言 1：位置 embedding 的 token 数与维度一致
    assert pos.shape == (expected_tokens, config.n_embedding), f"pos shape mismatch: {tuple(pos.shape)}"
    # 断言 2：RoPE 频率表的 token 数与维度一致
    assert rope.shape[0] == expected_tokens, f"rope token count mismatch: {tuple(rope.shape)}"
    assert rope.shape[1] == (encoder.head_dim // 2), f"rope dim mismatch: {tuple(rope.shape)}"
    # 断言 3：显式 d_image 与自动推断输出形状一致
    assert out_explicit.shape == out_infer.shape, f"explicit/infer output mismatch: {tuple(out_explicit.shape)} vs {tuple(out_infer.shape)}"
    # 断言 4：输出维度为 n_output_embed
    assert out_explicit.shape[-1] == config.n_output_embed, f"output dim mismatch: {tuple(out_explicit.shape)}"
    # 断言 5：输出数值有限（无 NaN/Inf）
    assert torch.isfinite(out_explicit).all().item(), "output contains NaN/Inf"

    logger.info("Smoke test passed")
    logger.info("device=%s dtype=%s", device, dtype)
    logger.info("pixels=%s", tuple(pixels.shape))
    logger.info("d_image=%s", tuple(d_image.shape))
    logger.info("pos=%s rope=%s out=%s", tuple(pos.shape), tuple(rope.shape), tuple(out_explicit.shape))


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()],
    )
    _smoke_test()
