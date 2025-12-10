# 隐式MLP注意力机制实现
# 基于Titans论文提出的隐式记忆MLP概念，将注意力机制与MLP结合
from __future__ import annotations

import torch
from torch import nn, cat, is_tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_map  # 用于处理嵌套数据结构的工具

from einops.layers.torch import Rearrange  # 用于张量维度重排的工具

from rotary_embedding_torch import RotaryEmbedding  # 旋转位置编码实现

# 辅助函数

def exists(v):
    """检查变量是否存在（不为None）"""
    return v is not None

# 主类定义

class ImplicitMLPAttention(Module):
    """隐式MLP注意力机制
    
    该模块将注意力机制与MLP结构相结合，实现了一种隐式的记忆MLP。
    它通过多个注意力层的链式连接来模拟MLP的计算过程，
    每个注意力层的键值对形成了MLP的隐式权重。
    """
    def __init__(
        self,
        dim,  # 输入特征维度
        mlp_hiddens: tuple[int, ...],  # MLP各层的隐藏维度，至少需要2层
        *,
        activation = nn.SiLU(),  # 激活函数，默认为SiLU
        heads = 8,  # 注意力头的数量
        talking_heads = True,  # 是否使用talking heads技术
        prenorm = True,  # 是否在注意力计算前进行归一化
        keys_rmsnorm = True  # 是否对键进行RMS归一化
    ):
        super().__init__()
        # 确保mlp_hiddens是一个元组且至少有2个元素
        assert isinstance(mlp_hiddens, tuple) and len(mlp_hiddens) >= 2
        # 解析MLP的输入、隐藏和输出维度
        dim_mlp_in, *dim_mlp_inner, dim_mlp_out = mlp_hiddens

        # 输入归一化
        self.norm = nn.RMSNorm(dim) if prenorm else nn.Identity()

        # 查询投影层，将输入映射到MLP输入维度×头数
        dim_query_inner = dim_mlp_in * heads
        self.to_queries = nn.Linear(dim, dim_query_inner, bias = False)

        # 旋转位置编码
        self.rotary_embed = RotaryEmbedding(min(mlp_hiddens))  # 使用最小的维度进行旋转编码

        # 键和值的投影层列表
        # 每个键值对形成一个隐式权重 (dim_key, dim_values)
        # 将它们链接起来就形成了TTT/Titans中的隐式MLP

        self.keys = ModuleList([])  # 键投影层列表
        self.key_norms = ModuleList([])  # 键归一化层列表
        self.values = ModuleList([])  # 值投影层列表

        # 为MLP的每一层创建键和值的投影层
        for dim_in, dim_out in zip(mlp_hiddens[:-1], mlp_hiddens[1:]):
            # 计算每一层的键和值的内部维度（考虑多头注意力）
            dim_keys_inner = dim_in * heads
            dim_values_inner = dim_out * heads

            # 创建键投影层
            keys = nn.Linear(dim, dim_keys_inner, bias = False)
            # 创建键归一化层
            key_norms = nn.RMSNorm(dim_in) if keys_rmsnorm else nn.Identity()
            # 创建值投影层
            values = nn.Linear(dim, dim_values_inner, bias = False)

            # 添加到模块列表
            self.keys.append(keys)
            self.key_norms.append(key_norms)
            self.values.append(values)

        # 激活函数
        self.activation = activation

        # talking head技术（来自Shazeer等人的工作）
        self.talking_heads = nn.Identity()

        # 如果启用talking heads且有隐藏层，则创建卷积层
        if talking_heads and len(dim_mlp_inner) > 0:
            self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            nn.init.dirac_(self.talking_heads.weight)  # 初始化为单位矩阵

        # 注意力头的拆分和合并操作
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)  # 将特征拆分为多个头
        self.merge_heads = Rearrange('b h n d -> b n (h d)')  # 将多个头的特征合并

        # 输出投影层，将MLP输出维度映射回原始输入维度
        self.to_out = nn.Linear(dim_mlp_out * heads, dim, bias = False)

    def forward(
        self,
        tokens,  # 输入序列，形状为 (batch, seq_len, dim)
        cache = None,  # 缓存的键值对，用于增量解码
        return_kv_cache = False  # 是否返回更新后的键值对缓存
    ):
        # 获取输入的批量大小、序列长度和设备
        batch, seq_len, device = *tokens.shape[:2], tokens.device

        # 输入归一化
        tokens = self.norm(tokens)

        # 生成查询向量
        queries = self.to_queries(tokens)

        # 生成键和值向量列表
        keys = [fn(tokens) for fn in self.keys]
        values = [fn(tokens) for fn in self.values]

        # 将查询、键和值拆分为多个注意力头
        queries, keys, values = tree_map(self.split_heads, (queries, keys, values))

        # 对键进行归一化
        keys = [norm(k) for norm, k in zip(self.key_norms, keys)]

        # 处理缓存（用于增量解码）
        if exists(cache):
            cache_keys, cache_values = cache
            # 将缓存的键值对与当前的键值对合并
            keys = [cat(args, dim = -2) for args in zip(cache_keys, keys)]
            values = [cat(args, dim = -2) for args in zip(cache_values, values)]

        # 注意力计算函数
        def attend(q, k, v):
            # 应用旋转位置编码
            q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)
            # 计算缩放点积注意力
            return F.scaled_dot_product_attention(q, k, v, is_causal = True)

        # 隐式记忆MLP计算
        out = queries  # 初始输出为查询向量

        # 遍历所有键值对层
        for i, (key, value) in enumerate(zip(keys, values), start = 1):
            is_last = i == len(keys)  # 检查是否是最后一层

            # 计算注意力
            out = attend(out, key, value)

            # 如果不是最后一层，应用talking heads和激活函数
            if not is_last:
                out = self.talking_heads(out)
                out = self.activation(out)

        # 合并注意力头
        out = self.merge_heads(out)
        # 输出投影
        out = self.to_out(out)

        # 根据需要返回输出和缓存
        if not return_kv_cache:
            return out

        return out, (keys, values)

# 3层隐式MLP注意力示例 - 64 -> 128 -> 128 -> 64，使用ReLU激活函数

if __name__ == '__main__':
    # 创建隐式MLP注意力实例
    implicit_mlp_attn = ImplicitMLPAttention(
        512,  # 输入特征维度
        (64, 128, 128, 64),  # MLP各层维度
        activation = nn.ReLU()  # 使用ReLU激活函数
    )

    # 创建随机输入张量
    tokens = torch.randn(1, 1024, 512)

    # 前向传播
    out, cache = implicit_mlp_attn(tokens)
    # 使用缓存进行增量解码
    out, cache = implicit_mlp_attn(tokens, cache = cache)

    # 验证输出形状与输入形状一致
    assert out.shape == tokens.shape
