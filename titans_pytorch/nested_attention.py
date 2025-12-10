"""
嵌套注意力机制实现

这个模块实现了Nested Attention机制，它是一种特殊的注意力架构，
使用多层嵌套的键值对来构建注意力权重，从而增强模型的表示能力。
"""

from __future__ import annotations

import torch
from torch import nn, cat, is_tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_map

from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

# 辅助函数

def exists(v):
    """检查变量是否存在
    
    Args:
        v: 要检查的变量
        
    Returns:
        bool: 如果变量不是None返回True，否则返回False
    """
    return v is not None

# 类定义

class NestedAttention(Module):
    """嵌套注意力机制
    
    实现了一种嵌套结构的注意力机制，通过三层键值对来构建注意力权重，
    增强了模型对复杂依赖关系的建模能力。
    
    Args:
        dim: 输入的维度
        dim_head: 每个注意力头的维度
        heads: 注意力头的数量
        prenorm: 是否在注意力前应用归一化
        keys_rmsnorm: 是否对键应用RMSNorm (参考: https://openreview.net/forum?id=HkztQWZfl2)
    """
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        prenorm = True,
        keys_rmsnorm = True # https://openreview.net/forum?id=HkztQWZfl2
    ):
        super().__init__()

        # 归一化层
        self.norm = nn.RMSNorm(dim) if prenorm else nn.Identity()

        # 计算内部维度
        dim_inner = dim_head * heads
        
        # 查询、键、值投影层
        self.to_queries = nn.Linear(dim, dim_inner, bias = False)

        # 旋转位置编码
        self.rotary_embed = RotaryEmbedding(dim_head)

        # 生成三层键和值
        self.to_keys = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_values = nn.Linear(dim, dim_inner * 3, bias = False)

        # 键的归一化层
        self.key_norms = ModuleList([nn.RMSNorm(dim_head) for _ in range(3)])
        self.nested_key_norm = nn.RMSNorm(dim_head)

        # 注意力头拆分和合并
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # 输出投影
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        tokens,
        cache = None,
        return_kv_cache = False
    ):
        """前向传播方法
        
        Args:
            tokens: 输入的token序列
            cache: 注意力缓存，用于加速推理
            return_kv_cache: 是否返回更新后的缓存
            
        Returns:
            如果return_kv_cache为True，返回输出和更新后的缓存
            否则只返回输出
        """
        batch, seq_len, device = *tokens.shape[:2], tokens.device

        # 归一化
        tokens = self.norm(tokens)

        # 生成查询
        queries = self.to_queries(tokens)

        # 生成三层键和值
        keys = self.to_keys(tokens).chunk(3, dim = -1)
        values = self.to_values(tokens).chunk(3, dim = -1)

        # 拆分注意力头
        queries, keys, values = tree_map(self.split_heads, (queries, keys, values))

        # 对所有键进行归一化
        keys = [norm(k) for norm, k in zip(self.key_norms, keys)]

        # 处理缓存
        if exists(cache):
            (cache_keys, cache_values), (cache_nested_keys, cache_nested_values) = cache
            # 合并缓存和当前键值对
            keys = [cat(args, dim = -2) for args in zip(cache_keys, keys)]
            values = [cat(args, dim = -2) for args in zip(cache_values, values)]

        # 注意力计算函数
        def attend(q, k, v):
            # 应用旋转位置编码
            q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)
            # 执行缩放点积注意力，使用因果掩码
            return F.scaled_dot_product_attention(q, k, v, is_causal = True)

        # 执行三层嵌套注意力
        nq, nk, nv = [attend(queries, key, value) for key, value in zip(keys, values)]

        # 对第二层注意力的输出进行归一化
        nk = self.nested_key_norm(nk)

        # 处理嵌套注意力的缓存
        if exists(cache):
            nk = cat((cache_nested_keys, nk), dim = -2)
            nv = cat((cache_nested_values, nv), dim = -2)

        # 执行最终的注意力计算
        out = attend(nq, nk, nv)

        # 合并注意力头
        out = self.merge_heads(out)

        # 输出投影
        out = self.to_out(out)

        # 返回结果
        if not return_kv_cache:
            return out

        return out, ((keys, values), (nk, nv))

# 测试代码
if __name__ == '__main__':
    # 创建嵌套注意力模型
    nested_attn = NestedAttention(512)

    # 生成随机输入
    tokens = torch.randn(1, 1024, 512)

    # 测试完整序列前向传播
    out1, cache = nested_attn(tokens)
    
    # 测试使用缓存的增量推理
    out2, cache = nested_attn(tokens[:, -1:], cache = cache)

    # 验证输出形状
    assert out1.shape == tokens.shape
    assert out2.shape == (1, 1, 512)
