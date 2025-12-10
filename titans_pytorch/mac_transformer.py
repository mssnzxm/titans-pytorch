# MAC Transformer 实现
# Memory-Augmented Context Transformer 是Titans论文中提出的一种增强上下文的Transformer架构

from __future__ import annotations
from typing import Callable

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm  # 进度条库

import torch
from torch import nn, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

# Flex Attention 是PyTorch提供的高效注意力实现
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)  # 如果有CUDA，编译加速
except ImportError:
    pass

def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding = False):
    """创建MAC块掩码
    
    为MAC Transformer创建注意力掩码，控制查询和键值对之间的可见性。
    
    参数:
        seq_len: 序列长度
        window_size: 窗口大小
        persist_mem_len: 持久化内存的长度
        sliding: 是否使用滑动窗口
    
    返回:
        block_mask: 注意力掩码
    """

    def create_mac_mask(_, __, q_idx, kv_idx):
        """内部函数，用于创建MAC掩码"""
        is_persist_mem = kv_idx < persist_mem_len  # 检查是否是持久化内存
        kv_without_mem = kv_idx - persist_mem_len
        causal_mask = q_idx >= kv_without_mem  # 因果掩码，确保只能看到过去的token

        if not sliding:
            # 块对角线掩码，只允许同一窗口内的注意力
            block_diagonal = (q_idx // window_size) == (kv_without_mem // window_size)
            causal_mask = causal_mask & block_diagonal
        else:
            # 滑动窗口掩码，允许看到过去window_size个token
            sliding_mask = (q_idx - kv_without_mem) <= window_size
            causal_mask = causal_mask & sliding_mask

        # 允许访问持久化内存或满足因果条件的token
        return is_persist_mem | (~is_persist_mem & causal_mask)

    # 创建并编译块掩码
    block_mask = create_block_mask(create_mac_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len + persist_mem_len, _compile = True)
    return block_mask

# Einstein notation 相关库
# 用于更简洁地操作张量维度

from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange

# 维度缩写说明：
# b - batch (批量)
# n - sequence (序列)
# h - heads (注意力头数)
# d - feature dimension (特征维度)

# 绝对和相对位置编码

from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding

# 超连接和注意力模块
# x-transformers提供的Attend模块能更好地处理不同长度的查询和键

from x_transformers.attend import Attend

from hyper_connections import get_init_and_expand_reduce_stream_functions

# 提出的神经内存模块

from titans_pytorch.neural_memory import NeuralMemory

# 常量定义

LinearNoBias = partial(Linear, bias = False)  # 无偏置的线性层

# 注意力中间结果的命名元组
AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))

# 辅助函数

def exists(v):
    """检查变量是否存在（不为None）"""
    return v is not None

def default(v, d):
    """如果变量存在则返回变量，否则返回默认值"""
    return v if exists(v) else d

def identity(t):
    """恒等函数，返回输入本身"""
    return t

def divisible_by(num, den):
    """检查第一个数是否能被第二个数整除"""
    return (num % den) == 0

def round_up_multiple(seq, mult):
    """将序列长度向上取整到指定倍数"""
    return ceil(seq / mult) * mult

def round_down_multiple(seq, mult):
    """将序列长度向下取整到指定倍数"""
    return seq // mult * mult

def pack_with_inverse(t, pattern):
    """将多个张量打包，并返回解包函数
    
    参数:
        t: 张量列表
        pattern: 打包模式
    
    返回:
        packed: 打包后的张量
        inverse: 解包函数
    """
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        """解包函数"""
        return unpack(out, packed_shape, default(inv_pattern, pattern))

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    """在指定维度上对张量进行填充
    
    参数:
        t: 输入张量
        pad: 填充大小，格式为 (左填充, 右填充)
        dim: 要填充的维度
        value: 填充值
    
    返回:
        填充后的张量
    """
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch = True,
    inverse_remove_pad = True
):
    """将序列填充并分割成段，并返回逆操作函数
    
    参数:
        seq: 输入序列，形状为 (batch, seq_len, dim)
        segment_len: 段长度
        fold_into_batch: 是否将段折叠到批次维度
        inverse_remove_pad: 逆操作时是否移除填充
    
    返回:
        seq: 处理后的序列
        inverse: 逆操作函数
    """
    batch, seq_len = seq.shape[:2]
    # 计算下一个序列长度的倍数
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    # 如果需要，进行填充
    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))

    # 如果需要，将段折叠到批次维度
    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n = segment_len)

    def inverse(out):
        """逆操作函数"""
        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b = batch)

        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]

        return out

    return seq, inverse

# 采样相关函数

def log(t, eps = 1e-20):
    """安全的对数函数，防止取对数时出现无穷大"""
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    """生成Gumbel噪声"""
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    """使用Gumbel-Softmax进行采样
    
    参数:
        t: 对数概率张量
        temperature: 温度参数，控制采样的随机性
    
    返回:
        采样的索引张量
    """
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim = -1, keepdim = True)

# Min-P过滤
# 参考论文: https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    """使用Min-P过滤对数概率
    
    参数:
        logits: 对数概率张量
        min_p: Min-P阈值
    
    返回:
        过滤后的对数概率张量
    """
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# 前馈网络和注意力模块

class GEGLU(Module):
    """门控线性单元变体 - GELU门控"""
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)  # 将输入分为两部分
        return F.silu(gate) * x  # 对门控部分应用SiLU激活，然后与另一部分相乘

def FeedForward(dim, mult = 4):
    """前馈网络
    
    参数:
        dim: 输入输出特征维度
        mult: 中间层的扩张因子
    
    返回:
        前馈网络模块
    """
    # 计算中间层维度，GEGLU通常使用 2/3 * mult * dim
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),  # 层归一化
        nn.Linear(dim, dim_inner * 2),  # 扩展维度
        GEGLU(),  # 门控激活
        nn.Linear(dim_inner, dim)  # 投影回原始维度
    )

class SegmentedAttention(Module):
    """分段注意力模块
    
    这是MAC Transformer中的核心注意力机制，支持分段处理、持久化内存和滑动窗口注意力。
    """
    def __init__(
        self,
        dim,  # 输入特征维度
        segment_len,  # 段长度
        num_persist_mem_tokens = 0,  # 持久化内存标记数量
        num_longterm_mem_tokens = 0,  # 长期内存标记数量
        dim_head = 64,  # 每个注意力头的维度
        heads = 8,  # 注意力头数量
        sliding = False,  # 是否使用滑动窗口注意力
        accept_value_residual = False,  # 是否接受值残差连接
        attend_kwargs: dict = dict(),  # Attend模块的额外参数
        use_flex_attn = False  # 是否使用Flex Attention
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)  # 层归一化

        dim_inner = dim_head * heads  # 内部维度（所有注意力头的总维度）

        self.rotary_emb = RotaryEmbedding(dim_head)  # 旋转位置编码

        self.attend = Attend(causal = True, **attend_kwargs)  # 因果注意力模块

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)  # 查询、键、值投影层
        self.to_out = LinearNoBias(dim_inner, dim)  # 输出投影层

        # 学习到的值混合权重（可选）
        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None

        self.segment_len = segment_len  # 段长度
        self.num_longterm_mem_tokens = num_longterm_mem_tokens  # 长期内存标记数量

        total_segment_len = segment_len + num_longterm_mem_tokens  # 总段长度
        self.total_segment_len = total_segment_len

        # 滑动窗口注意力设置
        self.sliding = sliding  # 滑动窗口注意力 - 重叠窗口的局部注意力通常更强

        # 注意力头的拆分和合并
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # 持久化内存参数
        self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))

        # Flex Attention相关设置

        assert not (use_flex_attn and not exists(flex_attention)), '需要最新版本的PyTorch并启用CUDA设备'
        self.use_flex_attn = use_flex_attn

        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens

    def forward_inference(
        self,
        token,
        cache,
        value_residual = None,
        output_gating = None,
    ):
        """推理时的单步前向传播
        
        用于在生成任务中逐词推理时调用，支持缓存机制以提高效率。
        """
        batch = token.shape[0]

        # 注意力计算前的归一化
        token = self.norm(token)

        # 生成查询、键、值
        q, k, v = self.to_qkv(token).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # 保存原始值（用于后续分析）
        orig_v = v

        # 应用学习到的值残差混合
        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(token)
            v = v.lerp(value_residual, mix)

        # 更新缓存
        ck, cv = cache
        k = cat((ck, k), dim = -2)
        v = cat((cv, v), dim = -2)
        next_cache = (k, v)

        # 应用旋转位置编码
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # 形状调整
        q, k, v = tuple(rearrange(t, 'b h n d -> b h n d') for t in (q, k, v))

        # 处理持久化内存的键和值
        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # 添加持久化内存
        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # 计算注意力
        out, _ = self.attend(q, k, v)

        # 合并注意力头
        out = self.merge_heads(out)

        # 输出投影
        out = self.to_out(out)

        # 应用输出门控（如果存在）
        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward_flex(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        output_gating = None,
        cache = None
    ):
        """使用Flex Attention的前向传播
        
        针对CUDA设备优化的前向传播方法，使用Flex Attention提高效率。
        """
        # 确保值残差和学习到的混合权重同时存在或不存在
        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        batch, seq_len = seq.shape[:2]

        # 注意力计算前的归一化
        seq = self.norm(seq)

        # 生成查询、键、值
        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # 保存原始值（用于后续分析）
        orig_v = v

        # 应用学习到的值残差混合
        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # 更新缓存
        next_cache = (k, v)

        # 处理持久化内存的键和值
        pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv b h n d', b = batch)

        # 应用旋转位置编码
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # 添加持久化内存
        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # 准备Flex Attention
        if not exists(flex_attn_fn):
            block_mask = create_mac_block_mask(seq_len, self.total_segment_len, self.num_persist_mem_tokens, self.sliding)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # 计算Flex Attention
        out = flex_attn_fn(q, k, v)

        # 合并注意力头
        out = self.merge_heads(out)

        # 输出投影
        out = self.to_out(out)

        # 应用输出门控（如果存在）
        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        disable_flex_attn = False,
        output_gating = None,
        cache = None
    ):
        """主前向传播方法
        
        根据输入情况选择不同的前向传播路径：
        1. 如果提供了缓存，使用forward_inference（推理模式）
        2. 如果在CUDA设备上且启用了Flex Attention，使用forward_flex
        3. 否则使用标准前向传播
        """
        is_inferencing = exists(cache)

        # 推理模式
        if is_inferencing:
            assert seq.shape[-2] == 1  # 推理时每次只处理一个token
            return self.forward_inference(seq, cache, value_residual, output_gating = output_gating)

        # CUDA设备上的Flex Attention模式
        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            return self.forward_flex(seq, value_residual, flex_attn_fn, output_gating = output_gating, cache = cache)

        # 确保值残差和学习到的混合权重同时存在或不存在
        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        segment_len, num_longterm_mem_tokens = self.segment_len, self.num_longterm_mem_tokens
        total_segment_len = segment_len + num_longterm_mem_tokens

        batch, seq_len = seq.shape[:2]

        # 自动填充到总段长度的倍数
        seq, inverse_segment = pad_and_segment_with_inverse(seq, total_segment_len, fold_into_batch = False)

        # 注意力计算前的归一化
        seq = self.norm(seq)

        # 生成查询、键、值
        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # 保存原始值（用于后续分析）
        orig_v = v

        # 应用学习到的值残差混合
        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # 更新缓存
        next_cache = tuple(map(inverse_segment, (k, v)))

        # 应用旋转位置编码
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # 折叠为段
        q, k, v = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = total_segment_len) for t in (q, k, v))

        # CPU上的滑动窗口处理
        attend_kwargs = dict()

        if self.sliding:
            # 调整形状以便处理滑动窗口
            k, v = tuple(rearrange(t, '(b w) ... -> b w ...', b = batch) for t in (k, v))
            # 填充
            k, v = tuple(pad_at_dim(t, (1, 0), value = 0., dim = 1) for t in (k, v))
            # 合并相邻窗口
            k = cat((k[:, :-1], k[:, 1:]), dim = -2)
            v = cat((v[:, :-1], v[:, 1:]), dim = -2)
            # 恢复形状
            k, v = tuple(rearrange(t, 'b w ... -> (b w) ...') for t in (k, v))

            # 处理滑动窗口掩码
            idx = torch.arange(seq.shape[-2], device = seq.device)
            q_idx = rearrange(idx, '(w n) -> w n', n = total_segment_len)
            k_idx = pad_at_dim(q_idx, (1, 0), dim = 0, value = -1e4)
            k_idx = cat((k_idx[:-1], k_idx[1:]), dim = -1)

            q_idx = rearrange(q_idx, 'w i -> w i 1')
            k_idx = rearrange(k_idx, 'w j -> w 1 j')

            sliding_mask = (q_idx - k_idx) <= total_segment_len
            sliding_mask = F.pad(sliding_mask, (self.num_persist_mem_tokens, 0), value = True)

            sliding_mask = repeat(sliding_mask, 'w i j -> (b w) 1 i j', b = batch)
            attend_kwargs.update(mask = sliding_mask)

        # 处理持久化内存的键和值
        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # 添加持久化内存
        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # 计算注意力
        out, _ = self.attend(q, k, v, **attend_kwargs)

        # 合并注意力头
        out = self.merge_heads(out)

        # 输出投影
        out = self.to_out(out)

        # 调整形状
        out = rearrange(out, '(b w) n d -> b (w n) d', b = batch)

        # 移除填充
        out = inverse_segment(out)

        # 应用输出门控（如果存在）
        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

# MAC transformer

class MemoryAsContextTransformer(Module):
    """记忆作为上下文的Transformer (MAC Transformer)
    
    这是MAC Transformer的核心实现，将神经网络记忆集成到Transformer架构中，
    通过在特定层引入神经内存模块来增强模型的长期依赖建模能力。
    """
    def __init__(
        self,
        *,
        num_tokens,  # 词汇表大小
        dim,  # 模型维度
        depth,  # 网络深度（层数）
        segment_len,  # 段长度
        neural_memory_segment_len = None,  # 神经内存段长度（默认与segment_len + num_longterm_mem_tokens相同）
        neural_mem_gate_attn_output = False,  # 是否使用神经内存门控注意力输出
        neural_memory_add_value_residual = False,  # 是否添加神经内存值残差
        num_longterm_mem_tokens = 0,  # 长期记忆标记数量
        num_persist_mem_tokens = 0,  # 持久化记忆标记数量
        neural_memory_batch_size = None,  # 神经内存批次大小
        neural_memory_qkv_receives_diff_views = False,  # 神经内存QKV是否接收不同视图
        dim_head = 64,  # 每个注意力头的维度
        heads = 8,  # 注意力头数量
        ff_mult = 4,  # 前馈网络维度乘数
        num_residual_streams = 4,  # 残差流数量
        neural_memory_model: Module | None = None,  # 神经内存模型
        neural_memory_kwargs: dict = dict(),  # 神经内存模型的额外参数
        neural_memory_layers: tuple[int, ...] | None = None,  # 哪些层使用神经内存
        use_flex_attn = False,  # 是否使用Flex Attention
        sliding_window_attn = False,  # 是否使用滑动窗口注意力
        neural_mem_weight_residual = False,  # 是否使用神经内存权重残差
        token_emb: Module | None = None,  # 自定义token嵌入层
    ):
        super().__init__()

        # 初始化token嵌入层
        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)
        self.token_emb = token_emb

        # 轴向位置编码 - 帮助模型区分段内和段间的位置信息
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim = dim, num_axial_dims = 2)

        # 长期记忆标记设置
        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        has_longterm_mems = num_longterm_mem_tokens > 0
        self.longterm_mems = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim) * 0.02)  # 可学习的长期记忆标记

        # 滑动窗口注意力设置
        self.sliding_window_attn = sliding_window_attn
        self.attn_window_size = segment_len + num_longterm_mem_tokens  # 注意力窗口大小

        # 超连接 - 初始化残差流的扩展和缩减函数
        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, add_stream_embed = True, disable = num_residual_streams == 1)

        self.layers = ModuleList([])  # 模型层列表

        # 神经内存段长度设置
        self.neural_memory_segment_len = default(neural_memory_segment_len, num_longterm_mem_tokens + segment_len)

        # 处理层索引和神经内存层设置
        layers = tuple(range(1, depth + 1))
        neural_memory_layers = default(neural_memory_layers, layers)  # 默认所有层都使用神经内存

        # 神经内存权重残差相关设置
        self.neural_mem_weight_residual = neural_mem_weight_residual
        is_first_neural_mem = True  # 标记是否为第一个神经内存层

        # 构建各层（内存、注意力和前馈网络）
        for layer in layers:
            is_first = layer == 1  # 是否为第一层

            # 初始化注意力模块
            attn = SegmentedAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                segment_len = segment_len,
                use_flex_attn = use_flex_attn,
                accept_value_residual = not is_first,
                num_longterm_mem_tokens = num_longterm_mem_tokens,
                num_persist_mem_tokens = num_persist_mem_tokens,
                sliding = sliding_window_attn
            )

            mem = None  # 神经内存模块
            mem_qkv_layer_selector = None  # QKV层选择器
            mem_hyper_conn = None  # 内存超连接

            # 如果当前层是神经内存层
            if layer in neural_memory_layers:
                # 初始化内存超连接
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual = not neural_mem_gate_attn_output)

                # 如果需要，初始化QKV层选择器
                if not is_first and neural_memory_qkv_receives_diff_views:
                    num_layer_choices = (layer - 1) * 4 + 1  # 每层有4个选择（attn输入、attn输出、ff输入、ff输出）加上当前残差流
                    mem_qkv_layer_selector = nn.Sequential(
                        nn.RMSNorm(dim),
                        nn.Linear(dim, 3 * num_layer_choices),
                        Rearrange('... (views layers) -> views ... layers', views = 3),
                        nn.Softmax(dim = -1)
                    )

                # 初始化神经内存模块
                mem = NeuralMemory(
                    dim = dim,
                    chunk_size = self.neural_memory_segment_len,
                    batch_size = neural_memory_batch_size,
                    model = deepcopy(neural_memory_model),
                    qkv_receives_diff_views = True,
                    accept_weight_residual = neural_mem_weight_residual and not is_first_neural_mem,
                    **neural_memory_kwargs
                )
                is_first_neural_mem = False  # 更新第一个神经内存层标记

            # 初始化前馈网络
            ff = FeedForward(dim = dim, mult = ff_mult)

            # 将当前层的模块添加到层列表
            self.layers.append(ModuleList([
                mem_hyper_conn,  # 内存超连接
                init_hyper_conn(),  # 注意力超连接
                init_hyper_conn(),  # 前馈网络超连接
                mem_qkv_layer_selector,  # QKV层选择器
                mem,  # 神经内存模块
                attn,  # 分段注意力模块
                ff,  # 前馈网络模块
            ]))

        # 最终归一化层
        self.norm = nn.RMSNorm(dim)

        # 输出到logits层
        self.to_logits = LinearNoBias(dim, num_tokens)

        # 是否使用神经内存门控注意力输出
        self.gate_attn_output = neural_mem_gate_attn_output

        # 用于辅助损失和设备设置的零张量
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # Flex Attention相关设置
        assert not (use_flex_attn and not exists(flex_attention)), '需要最新版本的PyTorch并启用CUDA设备'
        self.use_flex_attn = use_flex_attn

        self.num_persist_mem_tokens = num_persist_mem_tokens  # 持久化记忆标记数量

    def seq_index_is_longterm(
        self,
        seq_index
    ):
        """检查序列索引是否属于长期记忆
        
        Args:
            seq_index: 要检查的序列索引
            
        Returns:
            bool: 如果是长期记忆索引返回True，否则返回False
        """
        total_segment_len, segment_len = self.attn_window_size, self.segment_len
        return ((seq_index % total_segment_len + 1) - segment_len) > 0

    def seq_len_with_longterm_mem(
        self,
        seq_len
    ):
        assert seq_len > 0

        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return ((seq_len - 1) // segment_len) * num_mem + seq_len

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.5,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(
            min_p = 0.1,
        ),
        show_progress = True,
        use_cache = False
    ):
        """根据给定提示生成序列
        
        Args:
            prompt: 输入提示序列
            seq_len: 要生成的总序列长度
            temperature: 采样温度，控制输出多样性
            filter_fn: 采样过滤函数
            filter_kwargs: 过滤函数的额外参数
            show_progress: 是否显示进度条
            use_cache: 是否使用缓存加速采样
            
        Returns:
            生成的序列（不包含提示）
        """
        was_training = self.training
        self.eval()

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)  # 计算需要生成的token数量

        # 缓存设置：轴向位置编码、注意力和神经内存
        cache = None
        factorized_pos_emb = None

        # 预计算因子化位置编码
        if use_cache:
            seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)
            axial_dims = self.axial_pos_emb.maybe_derive_outer_dim(seq_len_with_mem, (self.neural_memory_segment_len,))
            factorized_pos_emb = self.axial_pos_emb(axial_dims, return_factorized = True)

        # 采样循环
        with tqdm.tqdm(total = sample_num_times, disable = not show_progress) as pbar:
            while out.shape[-1] < seq_len:
                # 前向传播获取logits
                logits, next_cache = self.forward(
                    out,
                    disable_flex_attn = True,  # 禁用Flex Attention以加速采样
                    cache = cache,  # 传递缓存
                    return_cache = True,  # 返回更新后的缓存
                    factorized_pos_emb = factorized_pos_emb  # 传递预计算的位置编码
                )

                if use_cache:
                    cache = next_cache  # 更新缓存

                if not exists(logits):
                    continue

                # 取最后一个token的logits
                logits = logits[:, -1]

                # 应用过滤函数和温度缩放
                logits = filter_fn(logits, **filter_kwargs)
                sample = gumbel_sample(logits, temperature = temperature)

                # 添加采样结果到输出
                out = torch.cat((out, sample), dim = -1)
                pbar.update(1)

        # 恢复模型训练状态
        self.train(was_training)

        # 返回生成的序列（不包含提示）
        return out[..., prompt_seq_len:]

    def forward(
        self,
        x,
        return_loss = False,
        return_loss_breakdown = False,
        disable_flex_attn = False,
        cache = None,
        return_cache = False,
        factorized_pos_emb = None
    ):
        """前馈传播方法
        
        Args:
            x: 输入序列
            return_loss: 是否返回损失值
            return_loss_breakdown: 是否返回损失分解
            disable_flex_attn: 是否禁用Flex Attention
            cache: 缓存用于推理
            return_cache: 是否返回更新后的缓存
            factorized_pos_emb: 预计算的因子化位置编码
            
        Returns:
            如果return_loss为True，返回损失值
            如果return_cache为True，返回logits和更新后的缓存
            否则返回logits
        """
        # 处理返回损失的情况
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # 提取各种参数和形状
        batch, seq_len, neural_mem_segment_len, segment_len, num_longterm_mem_tokens, attn_window_size = *x.shape, self.neural_memory_segment_len, self.segment_len, self.num_longterm_mem_tokens, self.attn_window_size
        seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)

        # Token嵌入
        x = self.token_emb(x)

        # 插入长期记忆
        x, inverse_segment = pad_and_segment_with_inverse(x, segment_len, inverse_remove_pad = False)
        mems = repeat(self.longterm_mems, 'n d -> b n d', b = x.shape[0])
        x, inverse_pack_mems = pack_with_inverse((x, mems), 'b * d')
        x = inverse_segment(x)

        # 裁剪多余的填充token
        x = x[:, :seq_len_with_mem]

        # 应用轴向位置编码 - 帮助模型区分段内和段间位置
        pos_emb = self.axial_pos_emb.forward_with_seq_len(seq_len_with_mem, (neural_mem_segment_len,), factorized = factorized_pos_emb)
        x = x + pos_emb

        # 准备Flex Attention
        use_flex_attn = x.is_cuda and self.use_flex_attn and not disable_flex_attn
        flex_attn_fn = None
        if use_flex_attn:
            # 创建MAC块掩码
            block_mask = create_mac_block_mask(seq_len_with_mem, self.attn_window_size, self.num_persist_mem_tokens, self.sliding_window_attn)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # KV缓存设置
        is_inferencing = exists(cache)
        if not exists(cache):
            cache = (seq_len_with_mem - 1, None, None)
        inference_seq_index, kv_caches, neural_mem_caches = cache

        kv_caches = iter(default(kv_caches, []))
        neural_mem_caches = iter(default(neural_mem_caches, []))

        next_kv_caches = []
        next_neural_mem_caches = []

        # 值残差和神经内存权重残差
        value_residual = None
        mem_weight_residual = None

        # 神经内存选择QKV输入的层列表
        mem_input_layers = []

        # 推理时，一次只处理一个token
        if is_inferencing:
            ind = inference_seq_index
            x = x[:, ind:(ind + 1)]

        # 扩展残差流用于超连接
        x = self.expand_streams(x)

        # 遍历所有层
        for mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, attn, ff in self.layers:
            retrieved = None
            attn_out_gates = None
            next_neural_mem_cache = None

            # 神经内存处理
            if exists(mem):
                # 内存输入
                mem_input, add_residual = mem_hyper_conn(x)

                # 处理QKV内存输入
                if not exists(mem_qkv_layer_selector):
                    # 如果没有层选择器，使用相同的输入
                    qkv_mem_input = stack((mem_input, mem_input, mem_input))
                else:
                    # 否则从可用层中选择
                    layers_to_choose_from = stack((mem_input, *mem_input_layers))
                    selected = mem_qkv_layer_selector(mem_input)
                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')

                # 通过神经内存
                retrieved, next_neural_mem_cache = mem.forward(
                    qkv_mem_input,
                    state = next(neural_mem_caches, None),
                    prev_weights = mem_weight_residual
                )

                # 更新内存权重残差
                if self.neural_mem_weight_residual:
                    mem_weight_residual = next_neural_mem_cache.updates

                # 处理注意力输出门控
                if self.gate_attn_output:
                    attn_out_gates = retrieved.sigmoid()
                else:
                    x = add_residual(retrieved)

            # 注意力处理
            attn_in, add_residual = attn_hyper_conn(x)
            mem_input_layers.append(attn_in)  # 添加到内存输入层列表

            # 执行注意力计算
            attn_out, (values, next_kv_cache) = attn(
                attn_in,
                value_residual = value_residual,
                disable_flex_attn = disable_flex_attn,
                flex_attn_fn = flex_attn_fn,
                output_gating = attn_out_gates,
                cache = next(kv_caches, None)
            )

            mem_input_layers.append(attn_out)
            value_residual = default(value_residual, values)
            x = add_residual(attn_out)  # 添加注意力残差

            # 更新缓存
            next_kv_caches.append(next_kv_cache)
            next_neural_mem_caches.append(next_neural_mem_cache)

            # 前馈网络处理
            ff_in, add_ff_residual = ff_hyper_conn(x)
            mem_input_layers.append(ff_in)
            ff_out = ff(ff_in)
            mem_input_layers.append(ff_out)
            x = add_ff_residual(ff_out)  # 添加前馈残差

        # 处理缓存
        if return_cache:
            next_kv_caches = stack([stack(kv_cache) for kv_cache in next_kv_caches])
            next_kv_caches = next_kv_caches[..., -attn_window_size:, :]  # 处理KV缓存长度
            kv_cache_length = next_kv_caches.shape[-2]
            
            # 如果不是滑动窗口注意力且缓存长度可被窗口大小整除，清空缓存
            if not self.sliding_window_attn and divisible_by(kv_cache_length, attn_window_size):
                next_kv_caches = next_kv_caches[..., 0:0, :]

            next_cache = (
                inference_seq_index + 1,
                next_kv_caches,
                next_neural_mem_caches
            )

            # 推理时处理长期记忆的情况
            is_longterm_mem = self.seq_index_is_longterm(inference_seq_index)
            if is_inferencing and is_longterm_mem:
                return None, next_cache

        # 合并残差流
        x = self.reduce_streams(x)

        # 移除记忆标记（非推理时）
        if not is_inferencing:
            x, inverse_segment = pad_and_segment_with_inverse(x, attn_window_size, inverse_remove_pad = False)
            x, _ = inverse_pack_mems(x)
            x = inverse_segment(x)
            x = x[:, :seq_len]  # 恢复原始序列长度

        # 归一化和转换为logits
        x = self.norm(x)
        logits = self.to_logits(x)

        # 返回结果
        if not return_loss:
            if not return_cache:
                return logits
            return logits, next_cache

        # 计算并返回损失
        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
