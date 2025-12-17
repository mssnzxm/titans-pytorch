from __future__ import annotations
from typing import Callable

import math
from functools import partial
from itertools import zip_longest
from collections import namedtuple

import torch
from torch import nn, stack, cat, is_tensor, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter, ParameterList, ParameterDict
from torch.func import functional_call, vmap, grad
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from tensordict import TensorDict

from assoc_scan import AssocScan

from titans_pytorch.memory_models import(
    MemoryMLP,
    ResidualNorm
)

import einx
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
h - heads
bh - batch and heads
n - sequence
d - feature dimension
c - intra-chunk
w - num memory network weight parameters
o - momentum orders
u - key / value updates - allowing a token to emit multiple key / values
"""

LinearNoBias = partial(Linear, bias = False)

# neural mem state related

NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates',
])

def mem_state_detach(
    state: NeuralMemState
):
    assert isinstance(state, NeuralMemState)
    state = tree_map(lambda t: t.detach() if is_tensor(t) else t, tuple(state))
    return NeuralMemState(*state)

# functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def identity(t):
    return t

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def safe_cat(inputs, dim = -2):
    inputs = tuple(filter(exists, inputs))

    if len(inputs) == 0:
        return None
    elif len(inputs) == 1:
        return inputs[0]

    return cat(inputs, dim = dim)

def is_empty_tensor(t):
    return t.numel() == 0

def dict_get_value_shapes(td):
    return [v.shape for k, v in td.items()]

def rearrange_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: rearrange(t, pattern, **kwargs))

def repeat_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: repeat(t, pattern, **kwargs))

def pair(v):
    return (v, v) if not isinstance(v, tuple) else v

def round_down_multiple(seq, mult):
    return seq // mult * mult

def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    if len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)

# softclamping gradients

def softclamp_max(t, max_value):
    half_max_value = max_value / 2
    return ((t / half_max_value).tanh() * half_max_value) + half_max_value

def softclamp_grad_norm(t, max_value):
    if is_empty_tensor(t):
        return t

    t, inverse = pack_one_with_inverse(t, 'bn *')

    norm = t.norm(dim = -1, keepdim = True)
    clamped_norm = softclamp_max(norm, max_value)

    t = t * (clamped_norm / norm)
    return inverse(t)

# spectral norming the surprise update w/ newton schulz matrix iter
# Keller Jordan et al. from OSS w/ nanogpt, now being used for two works, Atlas and 'TTT done right'

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    if t.ndim <= 3:
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = t.transpose(-1, -2)

    t, inv_pack = pack_one_with_inverse(t, '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t

    if should_transpose:
        t = t.transpose(-1, -2)

    return inv_pack(t)

# multi head rmsnorm

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

# chunk pooling

class AveragePool(Module):
    def __init__(
        self,
        chunk_size
    ):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)
        return reduce(x, 'b (n c) d -> b n d', 'mean', c = chunk_size)

class AttentionPool(Module):
    def __init__(
        self,
        dim,
        chunk_size
    ):
        """
        taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.to_attn_logits = nn.Linear(dim, dim)

        # default to average pool

        nn.init.zeros_(self.to_attn_logits.weight)
        nn.init.zeros_(self.to_attn_logits.bias)

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)

        x = rearrange(x, 'b (n c) d -> b n c d', c = chunk_size)

        attn_logits = self.to_attn_logits(x)

        attn = attn_logits.softmax(dim = -2)

        return reduce(x * attn, 'b n c d -> b n d', 'sum')

# main neural memory

def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr

def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)

class NeuralMemory(Module):
    """
    神经内存模块 - 实现可学习的动态内存存储与检索机制
    
    该模块是一个基于神经网络的内存系统，能够动态地存储和检索信息。它将内存表示为
    神经网络的权重，并通过学习的方式更新这些权重，从而实现高效的信息存储和检索。
    
    核心原理:
    - 将内存建模为神经网络(默认为MLP)的权重参数
    - 使用键值对(Key-Value)机制存储和检索记忆
    - 支持自适应学习率、动量和权重衰减等优化技术
    - 通过分块(chunk)处理实现高效的序列数据处理
    - 支持多头注意力机制，增强记忆表示能力
    
    主要功能:
    - store_memories: 将输入序列存储到神经内存中
    - retrieve_memories: 从神经内存中检索相关信息
    - 支持动量优化，提高内存更新的稳定性
    - 支持自适应学习率，根据输入动态调整内存更新强度
    - 支持多头结构，增强内存的表示能力
    
    应用场景:
    - 长序列建模任务
    - 记忆增强的语言模型
    - 需要长期依赖建模的任务
    - 持续学习和终身学习系统
    """
    def __init__(
        self,
        dim, # 输入特征维度
        chunk_size: int | tuple[int, int] = 1, # 分块大小，可分别指定检索和存储的分块大小
        batch_size = None, # 批处理大小
        dim_head = None, # 每个注意力头的维度，默认与dim相同
        heads = 1, # 注意力头的数量
        model: Module | None = None, # 内存模型，默认为MLP
        store_memory_loss_fn: Callable = default_loss_fn, # 存储记忆时使用的损失函数
        adaptive_step_transform: Callable | None = None, # 自适应学习率转换函数
        default_step_transform_max_lr = 1., # 默认学习率转换的最大学习率
        per_parameter_lr_modulation = False, # 是否允许外部网络控制每个权重矩阵的学习率
        max_mem_layer_modulation = 1., # 内存层调制的最大值
        per_head_learned_parameters = True, # 是否为每个注意力头学习独立的参数
        attn_pool_chunks = False, # 是否使用注意力池化来处理分块
        momentum = True, # 是否使用动量优化
        momentum_order = 1, # 动量的阶数
        learned_momentum_combine = False, # 是否学习动量的组合方式
        learned_combine_include_zeroth = False, # 是否在学习组合中包含零阶项
        num_kv_per_token = 1, # 每个标记可以进行的键值更新次数
        qkv_receives_diff_views = False, # 键值是否接收不同的视图
        pre_rmsnorm = True, # 是否在处理前使用RMSNorm归一化
        post_rmsnorm = False, # 是否在处理后使用RMSNorm归一化
        qk_rmsnorm = False, # 是否对查询和键使用RMSNorm归一化
        max_grad_norm: float | None = None, # 梯度范数的最大值，用于梯度裁剪
        use_accelerated_scan = False, # 是否使用加速扫描
        activation: Module | None = None, # 激活函数
        init_adaptive_step_bias = None, # 自适应步长的初始化偏置
        init_momentum_bias = None, # 动量的初始化偏置
        init_decay_bias = None, # 衰减因子的初始化偏置
        accept_weight_residual = False, # 是否接受权重残差
        spectral_norm_surprises = False, # 是否对更新前的惊喜值进行谱归一化
        gated_transition = False, # 是否使用门控过渡
        mem_model_norm_add_residual = True, # 是否在内存模型中添加残差连接和归一化
        default_model_kwargs: dict = dict(
            depth = 2, # 默认MLP模型的深度
            expansion_factor = 4. # 默认MLP模型的扩展因子
        )
    ):
        """
        初始化神经内存模块
        
        参数详解:
        - dim: 输入特征的维度
        - chunk_size: 分块处理的大小，可以是整数（检索和存储使用相同大小）或元组（分别指定检索和存储大小）
        - batch_size: 批处理大小，用于初始化内存状态
        - dim_head: 每个注意力头的特征维度，默认与dim相同
        - heads: 注意力头的数量，默认为1
        - model: 用于表示内存的神经网络模型，默认为MemoryMLP
        - store_memory_loss_fn: 存储记忆时使用的损失函数，默认为均方误差
        - adaptive_step_transform: 自适应学习率的转换函数，用于将模型输出转换为学习率
        - per_parameter_lr_modulation: 是否允许为每个权重矩阵单独控制学习率
        - momentum: 是否使用动量优化来稳定内存更新
        - num_kv_per_token: 每个输入标记可以生成的键值对数量
        - pre_rmsnorm: 是否在处理前对输入进行RMSNorm归一化
        - max_grad_norm: 梯度裁剪的最大范数，用于稳定训练
        """
        super().__init__()
        dim_head = default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)

        self.retrieve_chunk_size, self.store_chunk_size = pair(chunk_size)

        # batch size

        if exists(batch_size):
            assert divisible_by(batch_size, self.store_chunk_size)

        self.batch_size = batch_size

        # associative scan

        self.assoc_scan = AssocScan(use_accelerated = use_accelerated_scan)

        # key values receiving different views

        self.qkv_receives_diff_views = qkv_receives_diff_views

        # norms

        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()

        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()

        # maybe multi-headed

        dim_inner = dim_head * heads

        self.heads = heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_kv_heads = Rearrange('b n (h u d) -> b h (n u) d', h = heads, u = num_kv_per_token)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        self.retrieve_gate = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if heads > 1 else None

        # memory model

        if not exists(model):
            model = MemoryMLP(dim_head, **default_model_kwargs)

        # validate memory model

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        test_shape = (3, 2, dim_head)

        with torch.no_grad():
            try:
                test_input = torch.randn(test_shape)
                mem_model_output = model(test_input)
            except:
                raise RuntimeError(f'memory model unable to accept a tensor of shape {test_shape}')

            assert mem_model_output.shape == test_shape, 'output of memory model needs to be same shape as input'

        # the memory is the weights of the model

        if mem_model_norm_add_residual:
            model = ResidualNorm(dim = dim_head, model = model)

        self.memory_model = model

        mem_model_params = dict(model.named_parameters())

        self.num_memory_parameter_tensors = len(mem_model_params)

        self.memory_model_parameter_names = [*mem_model_params.keys()]

        memory_model_parameters = [*mem_model_params.values()]

        if per_head_learned_parameters:
            memory_model_parameters = [repeat(p, '... -> h ...', h = heads) for p in memory_model_parameters]

        self.init_weight_shape = [p.shape for p in memory_model_parameters]

        self.memory_model_parameters = ParameterList(memory_model_parameters)
        self.per_head_learned_parameters = per_head_learned_parameters

        # the chunk size within the paper where adaptive step, momentum, weight decay are shared

        self.chunk_size = chunk_size

        # prepare function for per sample gradients from model above, using torch.func

        def forward_and_loss(params, inputs, loss_weights, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            weighted_loss = loss * loss_weights
            return weighted_loss.sum(), loss

        # two functions

        grad_fn = grad(forward_and_loss, has_aux = True)

        self.per_sample_grad_fn = vmap(grad_fn, in_dims = (0, 0, 0, 0))

        # queries for retrieving from the model

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), activation)

        # keys and values for storing to the model

        assert num_kv_per_token > 0

        self.to_keys = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        self.to_values = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        self.store_memory_loss_fn = store_memory_loss_fn

        self.num_kv_per_token = num_kv_per_token

        # `chunk_size` refers to chunk size used for storing to memory model weights

        chunk_size = self.store_chunk_size

        # whether to use averaging of chunks, or attention pooling

        assert not (attn_pool_chunks and chunk_size == 1), '`attn_pool_chunks` cannot be set to True if `chunk_size` is set to 1'

        if not attn_pool_chunks:
            self.reduce_to_chunk_rep = AveragePool(chunk_size = chunk_size)
        else:
            self.reduce_to_chunk_rep = AttentionPool(dim, chunk_size = chunk_size)

        # learned adaptive learning rate

        self.to_adaptive_step = Sequential(
            nn.Linear(dim, heads * num_kv_per_token),
            Rearrange('b n (h u) -> (b h) (n u)', u = num_kv_per_token)
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(default_adaptive_step_transform, max_lr = default_step_transform_max_lr)

        self.adaptive_step_transform = adaptive_step_transform

        # momentum related

        self.to_momentum = Sequential(
            nn.Linear(dim, heads * momentum_order),
            Rearrange('b n (h o) -> o (b h) n 1', o = momentum_order)
        ) if momentum else None

        self.momentum_order = momentum_order
        self.to_learned_momentum_combine = None

        if learned_momentum_combine:
            assert momentum
            assert momentum_order > 1, 'only second order momentum allowed for now, but may allow learned combination of zeroth'

            if learned_combine_include_zeroth:
                momentum_order += 1

            self.to_learned_momentum_combine = Sequential(
                nn.Linear(dim, heads * momentum_order),
                Rearrange('b n (h o) -> o (b h) n', h = heads),
                nn.Softmax(dim = 0),
            )

            self.learned_combine_include_zeroth = learned_combine_include_zeroth

        # per layer learning rate modulation

        self.to_layer_modulation = Sequential(
            nn.Linear(dim, heads * self.num_memory_parameter_tensors),
            Rearrange('b n (h w) -> w (b h) n', h = heads),
            nn.Sigmoid()
        ) if per_parameter_lr_modulation else None

        self.max_mem_layer_modulation = max_mem_layer_modulation

        # learned weight residual

        self.to_learned_weight_residual_mix = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n'),
            nn.Sigmoid()
        ) if accept_weight_residual else None

        # allow for softclamp the gradient norms for storing memories

        self.max_grad_norm = max_grad_norm

        # spectral norming the surprises before update, a la Muon from Jordan et al.

        self.spectral_norm_surprises = spectral_norm_surprises

        # weight decay factor

        self.to_decay_factor = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        )

        # learned transition, as seeing instability when decreasing neural mem batch size
        # perhaps it can slowly learn to adjust from early residual to fully transitioning to new weights every batch size

        self.transition_gate = nn.Parameter(tensor(-5.)) if gated_transition else None

        # inits

        if exists(init_adaptive_step_bias):
            linear = self.to_adaptive_step[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_adaptive_step_bias)

        if exists(init_momentum_bias):
            linear = self.to_momentum[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_momentum_bias)

        if exists(init_decay_bias):
            linear = self.to_decay_factor[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_decay_bias)

        # maybe use accelerated scan

        self.use_accelerated_scan = use_accelerated_scan

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def memory_model_parameter_dict(self):
        """
        获取内存模型参数的字典表示
        
        返回一个TensorDict对象，包含内存模型的所有参数，其中键是参数名称，值是对应的参数张量。
        这种表示方式便于使用torch.func进行函数式调用和梯度计算。
        
        返回值:
        - TensorDict: 包含内存模型所有参数的字典
        """
        return TensorDict(dict(zip(self.memory_model_parameter_names, self.memory_model_parameters)))

    def init_weights(
        self,
        batch,
    ):
        """
        初始化内存模型的权重
        
        根据批处理大小和注意力头配置，初始化内存模型的权重。支持为每个注意力头
        学习独立的参数或共享参数。
        
        参数:
        - batch: 批处理大小
        
        返回值:
        - TensorDict: 初始化后的权重字典，包含内存模型的所有参数
        
        逻辑:
        - 如果per_head_learned_parameters为True，为每个注意力头创建独立的参数副本
        - 否则，为所有注意力头和批处理创建共享参数的副本
        """
        if self.per_head_learned_parameters:
            weights = repeat_dict_values(self.memory_model_parameter_dict, 'h ... -> (b h) ...', b = batch)
        else:
            weights = repeat_dict_values(self.memory_model_parameter_dict, '... -> bh ...', bh = batch * self.heads)

        return weights

    def init_momentum(
        self,
        batch,
    ):
        """
        初始化动量状态
        
        创建与内存模型参数结构相同的零张量字典，用于存储动量信息。动量是一种优化技术，
        可以使内存更新更加稳定，通过累积过去的梯度方向来平滑当前的更新。
        
        参数:
        - batch: 批处理大小
        
        返回值:
        - TensorDict: 初始化后的动量状态字典，包含与内存模型参数结构相同的零张量
        
        逻辑:
        - 创建内存模型参数的零副本
        - 根据per_head_learned_parameters配置和动量阶数，扩展零张量到适当的形状
        - 返回初始化后的动量状态
        """
        zeros = self.memory_model_parameter_dict.clone().zero_()

        if self.per_head_learned_parameters:
            zeros = repeat_dict_values(zeros, 'h ... -> o (b h) ...', b = batch, o = self.momentum_order)
        else:
            zeros = repeat_dict_values(zeros, '... -> o bh ...', bh = batch * self.heads, o = self.momentum_order)

        return zeros

    def store_memories(
        self,
        seq,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        seq_index = 0,
        prev_weights = None,
        mask: Tensor | None = None,
        return_surprises = True
    ):
        """
        将输入序列存储到神经内存中
        
        该方法是神经内存模块的核心功能之一，负责将输入序列编码为键值对，并使用这些
        键值对更新内存模型的权重。它实现了一个基于梯度的内存更新机制，通过计算预测
        损失并反向传播来更新内存。
        
        参数:
        - seq: 输入序列，形状为(batch, seq_len, dim)或(views, batch, seq_len, dim)
        - weights: 内存网络的权重字典，如果为None则初始化新权重
        - past_state: 过去的状态，包含权重和动量信息的元组
        - seq_index: 当前序列索引，用于跟踪处理进度
        - prev_weights: 上一层的权重，用于影响当前层的surprise计算
        - mask: 存储掩码，用于控制哪些位置的记忆被存储
        - return_surprises: 是否返回surprises信息，包含损失和自适应学习率
        
        返回值:
        - updates: 更新后的内存网络权重字典
        - next_store_state: 下一个存储状态，包含更新后的权重和动量
        - surprises: (可选)包含unweighted_mem_model_loss和adaptive_lr的元组
        
        核心实现步骤:
        1. 输入预处理：将序列裁剪为chunk_size的整数倍，只处理完整的分块
        2. 权重初始化：如果没有提供权重，则使用init_weights初始化
        3. 特征提取：通过线性层将输入序列转换为键和值
        4. 自适应学习率计算：根据输入序列动态计算学习率
        5. 动量更新：使用动量优化技术平滑内存更新
        6. 梯度计算：通过functional_call和vmap计算内存模型的梯度
        7. 权重更新：使用计算得到的梯度和自适应学习率更新内存权重
        8. 状态管理：返回更新后的权重和下一个存储状态
        
        关键技术点:
        - 分块处理：将长序列分为多个分块，提高处理效率
        - 自适应学习率：根据输入内容动态调整内存更新强度
        - 动量优化：使用动量技术提高内存更新的稳定性
        - 多头注意力：支持多头结构，增强内存表示能力
        """
        if self.qkv_receives_diff_views:
            _, batch, seq_len = seq.shape[:3]
        else:
            batch, seq_len = seq.shape[:2]

        # 获取配置参数
        heads, chunk_size, num_updates = self.heads, self.store_chunk_size, self.num_kv_per_token

        # 将序列裁剪为chunk_size的整数倍
        # 只有完整的chunk才能为下一个chunk提供记忆
        round_down_seq_len = round_down_multiple(seq_len, chunk_size)
        num_chunks = round_down_seq_len // chunk_size

        seq, remainder = seq[..., :round_down_seq_len, :], seq[..., round_down_seq_len:, :]

        next_seq_len_index = seq_index + round_down_seq_len

        # 如果没有提供权重，则初始化权重
        if not exists(weights):
            weights = self.init_weights(batch)

        weights = TensorDict(weights)

        # 允许前一层的神经内存影响当前层的surprise计算
        weights_for_surprise = repeat_dict_values(weights, 'b ... -> b n ...', n = num_chunks)

        # 对输入序列进行归一化
        seq = self.store_norm(seq)

        # 处理来自超连接的不同序列的键和值
        values_seq = seq

        if self.qkv_receives_diff_views:
            seq, values_seq = seq

        # 为内存网络的优化推导出学习的超参数
        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        # 将序列减少到chunk表示
        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size)

        # 计算衰减因子
        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        # 检查是否需要层学习率调制和动量
        need_layer_lr_mod = exists(self.to_layer_modulation) and num_chunks > 0
        has_momentum = exists(self.to_momentum)

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()

            learned_combine = exists(self.to_learned_momentum_combine)

            if learned_combine:
                combine_momentums = self.to_learned_momentum_combine(chunked_seq)

        if need_layer_lr_mod:
            layer_lr_mod = self.to_layer_modulation(chunked_seq) * self.max_mem_layer_modulation

        # 生成键和值
        keys = self.to_keys(seq)
        values = self.to_values(values_seq)

        # 多头处理
        keys, values = map(self.split_kv_heads, (keys, values))

        # 对键进行RMS归一化
        keys = self.k_norm(keys)

        # 处理分块
        keys, values = tuple(rearrange(t, 'b h (n c u) d -> (b h n) (c u) d', c = chunk_size, u = num_updates) for t in (keys, values))

        # 调整自适应学习率的形状
        adaptive_lr = rearrange(adaptive_lr, 'b (n c u) -> (b n) (c u)', c = chunk_size, u = num_updates)

        # 应用存储记忆掩码，如果为False，则将该位置的学习率设置为0
        if exists(mask):
            mask = mask[..., :round_down_seq_len]
            mask = repeat(mask, 'b (n c) -> (b h n) (c u)', h = heads, u = num_updates, c = chunk_size)

            adaptive_lr = torch.where(mask, adaptive_lr, 0.)

        # 断言检查learned_weight_residual_mix和prev_weights的存在性是否一致
        assert xnor(exists(self.to_learned_weight_residual_mix), exists(prev_weights))

        # 添加前一层的权重
        if exists(prev_weights):
            start_index = math.ceil(seq_index / chunk_size)
            end_index = start_index + num_chunks

            prev_weights = prev_weights.apply(lambda t: t[:, start_index:end_index])

            if exists(self.to_learned_weight_residual_mix) and num_chunks > 0:
                mix = self.to_learned_weight_residual_mix(chunked_seq)
                mix = rearrange(mix, 'b h n -> (b h) n')
                prev_weights = prev_weights.apply(lambda t: einx.multiply('bh n, bh n ... -> bh n ...', mix, t))

            weights_for_surprise = weights_for_surprise + prev_weights

        # 展平批处理和时间维度（如果surprise依赖于前一层的内存模型）
        weights_for_surprise = rearrange_dict_values(weights_for_surprise, 'b n ... -> (b n) ...')

        # 获取梯度和额外的辅助损失（用于通过基础神经内存模块中的qkv投影进行反向传播）
        grads, unweighted_mem_model_loss = self.per_sample_grad_fn(dict(weights_for_surprise), keys, adaptive_lr, values)

        grads = TensorDict(grads)

        # 调整adaptive_lr和unweighted_mem_model_loss的形状
        adaptive_lr = rearrange(adaptive_lr, '(b h n) c -> b h (n c)', b = batch, h = heads)
        unweighted_mem_model_loss = rearrange(unweighted_mem_model_loss, '(b h n) c -> b h (n c)', b = batch, h = heads)

        # 软钳位梯度范数
        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        # 恢复批处理和序列维度
        grads = rearrange_dict_values(grads, '(b n) ... -> b n ...', b = batch * heads)

        # 每层调制
        if need_layer_lr_mod:
            grads = TensorDict({name: einx.multiply('b h, b h ... -> b h ...', layer_lr_mod, t) for layer_lr_mod, (name, t) in zip(layer_lr_mod, grads.items())})

        # 负梯度，自适应学习率已经作为损失权重应用
        surprises = grads.mul(-1)

        # past states

        if not exists(past_state):
            # minibatch_init_weight corresponds to W0 in figure 7 of TTT paper

            minibatch_init_weight = weights
            init_momentum = self.init_momentum(batch)

            past_state = (minibatch_init_weight, init_momentum)

        past_last_update, past_last_momentum = past_state

        # early return if sequence length less than chunk size

        if num_chunks == 0:
            updates = rearrange_dict_values(weights, 'bh ... -> bh 1 ...')
            next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, past_state, updates)

            output = (updates, next_store_state)

            if not return_surprises:
                return output

            return (*output, (unweighted_mem_model_loss, adaptive_lr))

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates

        updates = TensorDict()

        next_last_update = TensorDict()
        next_last_momentum = TensorDict()

        for (param_name, surprise), (_, last_update) in zip(surprises.items(), past_last_update.items()):

            update = surprise

            # derive momentum with associative scan - eq (10)

            if has_momentum:
                momentum = surprise

                momentums = [] # stores all momentum orders starting with first, to generalize to Nth order momentum

                last_momentum = past_last_momentum[param_name]

                # go from first order momentum all the way to the Nth

                for one_adaptive_momentum, one_last_momentum in zip_longest(adaptive_momentum, last_momentum):
                    momentum = self.assoc_scan(one_adaptive_momentum, momentum, prev = one_last_momentum) # momentum is S / surprise in the paper

                    momentums.append(momentum)

                momentums = stack(momentums)

                next_last_momentum[param_name] = momentums[:, :, -1] # momentums shape is Float['o bh n 1']

                if learned_combine and self.learned_combine_include_zeroth:
                    # add the original surprise if learned combination of momentums
                    momentums = cat((rearrange(surprise, '... -> 1 ...'), momentums), dim = 0)

                if not learned_combine:
                    update = momentums[-1]
                else:
                    update = einsum(combine_momentums, momentums, 'o b n, o b n ... -> b n ...')

            # maybe spectral norm surprises

            if self.spectral_norm_surprises:
                update = newtonschulz5(update)

            # use associative scan again for learned forgetting (weight decay) - eq (13)

            update = self.assoc_scan(1. - decay_factor, update, prev = last_update, remove_prev = False)

            updates[param_name] = update
            next_last_update[param_name] = update[:, -1]

        # determine next state for the storing of memories

        next_state = (next_last_update, next_last_momentum)

        next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, next_state, updates)

        # return updates to neural memory at all chunked timesteps + neural mem cache / state to be fed back

        if not return_surprises:
            return updates, next_store_state

        return updates, next_store_state, (unweighted_mem_model_loss, adaptive_lr)

    def retrieve_memories(
        self,
        seq,
        weights: dict[str, Tensor] | None = None,
    ):
        """
        从神经内存中检索相关信息
        
        该方法是神经内存模块的核心功能之一，负责使用输入序列查询神经内存，
        并返回与查询相关的记忆信息。它实现了基于注意力机制的记忆检索过程。
        
        参数:
        - seq: 输入序列，形状为(batch, seq_len, dim)，用于查询记忆
        - weights: 内存网络的权重字典，包含已存储的记忆信息
        
        返回值:
        - retrieved_values: 检索到的记忆值，形状与输入序列一致
        
        核心实现步骤:
        1. 输入预处理：根据分块大小对序列进行填充，确保可以完整处理
        2. 查询生成：通过线性层将输入序列转换为查询向量
        3. 多头处理：将查询向量分割为多个注意力头进行并行处理
        4. 记忆检索：使用functional_call调用内存模型，通过查询访问存储的记忆
        5. 结果后处理：合并多头结果并移除填充，返回最终检索值
        
        关键技术点:
        - 自动推断单token解码场景，优化处理效率
        - 动态填充机制，确保序列长度为分块大小的整数倍
        - 多头注意力机制，增强记忆检索能力
        - 支持RMSNorm归一化，稳定训练和推理过程
        - 与MLP内存模型的无缝集成，支持复杂的记忆表示
        """
        chunk_size = self.retrieve_chunk_size

        # 检查权重是否具有扩展形状
        weights_have_expanded_shape = dict_get_value_shapes(weights) != self.init_weight_shape

        batch, seq_len = seq.shape[:2]

        # 自动推断是否为单个token解码（如果只有1组权重和1个token）
        is_one_token = seq_len == 1
        is_one_weight = (not weights_have_expanded_shape) or next(iter(weights.values())).shape[1] == 1

        is_single_token_decode = is_one_token and is_one_weight

        if is_single_token_decode:
            chunk_size = 1

        # 与分块处理相关的填充
        need_pad = chunk_size > 1 or not is_one_weight

        if need_pad:
            seq = pad_at_dim(seq, (1, 0), dim = 1)

        seq_len_plus_one = seq.shape[-2]

        # 将序列长度向上舍入到chunk_size的整数倍
        next_seq_len = round_up_multiple(seq_len_plus_one, chunk_size)

        # 计算需要填充的长度并进行填充
        padding = next_seq_len - seq_len_plus_one
        seq = pad_at_dim(seq, (0, padding), dim = 1)

        # 内存模型的参数存储键/值的记忆
        # 当MLP只有1个权重矩阵时，它等价于线性注意力文献中的`kv`快速权重记忆（记忆的获取是q @ (kv)）/ schmidhuber的论文

        weights = TensorDict(weights)

        # 对序列进行归一化
        seq = self.retrieve_norm(seq)

        # 从序列生成查询
        queries = self.to_queries(seq)

        # 多头处理
        queries = self.split_heads(queries)

        # 对查询进行RMS归一化
        queries = self.q_norm(queries)

        # 从内存模型获取值
        if weights_have_expanded_shape:
            weights = rearrange_dict_values(weights, 'b n ... -> (b n) ...')

        # 调整查询的形状以适应分块处理
        queries = rearrange(queries, 'b h (n c) d -> (b h n) c d', c = chunk_size)

        # 前向函数调用
        values = functional_call(self.memory_model, dict(weights), queries)

        # 重构批处理维度
        values = rearrange(values, '(b h n) c d -> b h (n c) d', b = batch, h = self.heads)

        # 多头RMS归一化
        values = self.multihead_rmsnorm(values)

        # 可选的门控
        if exists(self.retrieve_gate):
            values = values * self.retrieve_gate(seq)

        # 合并头并组合
        values = self.merge_heads(values)
        values = self.combine_heads(values)

        # 恢复原始序列长度，移除填充
        if need_pad:
            values = values[:, 1:]

        return values[:, :seq_len]

    def forward(
        self,
        seq,
        store_seq = None,
        state: NeuralMemState | None = None,
        detach_mem_state = False,
        prev_weights = None,
        store_mask: Tensor | None = None,
        return_surprises = False,
        ttt_batch_size: int | None = None
    ):
        """
        神经内存的前向传播接口
        
        该方法是NeuralMemory类的主要外部接口，协调记忆的存储和检索过程。
        它接收输入序列，根据当前状态更新内存，并返回检索到的记忆值和新的状态。
        
        参数:
        - seq: 输入序列，形状为(batch, seq_len, dim)或(views, batch, seq_len, dim)
        - store_seq: 用于存储的序列，如果为None则使用seq
        - state: 神经内存状态，包含序列索引、权重、缓存、过去状态和更新信息
        - detach_mem_state: 是否分离内存状态，用于控制梯度流
        - prev_weights: 上一层的权重，用于影响当前层的surprise计算
        - store_mask: 存储掩码，用于控制哪些位置的记忆被存储
        - return_surprises: 是否返回surprises信息，包含损失和自适应学习率
        - ttt_batch_size: TTT批处理大小，用于控制权重更新的频率
        
        返回值:
        - retrieved_values: 检索到的记忆值，形状与输入序列一致
        - next_state: 更新后的神经内存状态
        - surprises: (可选)包含unweighted_mem_model_loss和adaptive_lr的元组
        
        核心实现步骤:
        1. 输入预处理：处理单token输入和多视图输入场景
        2. 状态初始化：如果没有提供状态，则创建新状态
        3. 存储序列处理：合并缓存的存储序列和当前存储序列
        4. 序列分块：根据批处理大小和序列索引确定更新边界和分块大小
        5. 记忆存储：循环处理每个序列分块，调用store_memories更新内存
        6. 权重更新：在批处理边界处更新内存模型的权重
        7. 记忆检索：调用retrieve_memories从更新后的内存中检索信息
        8. 后处理：根据需要分离内存状态并返回结果
        
        关键技术点:
        - 支持单token解码和批量处理两种模式
        - 自动处理多视图输入，为查询和键值提供不同的视图
        - 智能状态管理，支持序列的连续处理
        - 批处理边界检测，在适当的时机更新内存权重
        - 支持门控过渡机制，平滑调整内存更新强度
        - 与store_memories和retrieve_memories方法的无缝集成
        """
        is_multi_input = self.qkv_receives_diff_views

        # 处理单个token输入
        if seq.ndim == 2 or (is_multi_input and seq.ndim == 3):
            seq = rearrange(seq, '... b d -> ... b 1 d')

        is_single_token = seq.shape[-2] == 1

        # 如果qkv接收不同的视图
        if is_multi_input:
            retrieve_seq, seq = seq[0], seq[1:]
        else:
            retrieve_seq = seq

        # 处理前一个状态的初始化
        if not exists(state):
            state = (0, None, None, None, None)

        seq_index, weights, cache_store_seq, past_state, updates = state

        # 获取用于存储的序列
        store_seq = default(store_seq, seq)

        # 处理缓存
        if exists(cache_store_seq):
            store_seq = safe_cat((cache_store_seq, store_seq))

        # compute split sizes of sequence
        # for now manually update weights to last update at the correct boundaries

        store_seq_len, chunk_size, batch_size = store_seq.shape[-2], self.chunk_size, default(ttt_batch_size, self.batch_size)

        need_update_weights = exists(batch_size)

        # determine split sizes and when to update

        if need_update_weights:
            update_after_final_store = divisible_by(seq_index + store_seq_len, batch_size)

            seq_range = torch.arange(store_seq_len) + seq_index + 1
            batch_boundary = divisible_by(seq_range, batch_size)

            indices = seq_range[batch_boundary] - seq_index

            indices = F.pad(indices, (1, 0), value = 0)

            if indices[-1] != store_seq_len:
                indices = F.pad(indices, (0, 1), value = store_seq_len)

            split_sizes = (indices[1:] - indices[:-1]).tolist()

            assert sum(split_sizes) == store_seq_len
        else:
            split_sizes = (store_seq_len,)
            update_after_final_store = False

        # accumulate updates

        updates = None

        def accum_updates(past_updates, future_updates):
            if not exists(past_updates):
                return future_updates

            return TensorDict({param_name: cat((past_update[:, :-1], future_update), dim = 1) for (param_name, past_update), (_, future_update) in zip(past_updates.items(), future_updates.items())})

        # loop through chunks of store sequences

        store_seqs = store_seq.split(split_sizes, dim = -2)

        if exists(store_mask):
            store_masks = store_mask.split(split_sizes, dim = -1)
        else:
            store_masks = (None,) * len(split_sizes)

        # whether to allow network to slowly adjust from initial weight throughout (residual path) to fully updating weights every batch

        surprises = (None, None)
        gate = None

        if exists(self.transition_gate):
            gate = self.transition_gate.sigmoid()

        for ind, (store_seq_chunk, maybe_store_mask) in enumerate(zip(store_seqs, store_masks)):
            is_last = ind == (len(store_seqs) - 1)

            # store

            next_updates, next_neural_mem_state, chunk_surprises = self.store_memories(
                store_seq_chunk,
                weights,
                seq_index = seq_index,
                past_state = past_state,
                prev_weights = prev_weights,
                mask = maybe_store_mask,
                return_surprises = True
            )

            weights = next_neural_mem_state.weights
            seq_index = next_neural_mem_state.seq_index
            past_state = next_neural_mem_state.states

            updates = accum_updates(updates, next_updates)

            surprises = tuple(safe_cat(args, dim = -1) for args in zip(surprises, chunk_surprises))

            if is_last and not update_after_final_store:
                continue

            # update weights once batch size is fulfilled

            last_update, last_momentum = past_state

            if exists(gate):
                last_update = TensorDict({param_name: one_weight.lerp(one_last_update, gate) for (param_name, one_weight), (_, one_last_update) in zip(weights.items(), last_update.items())})

            past_state = (last_update, last_momentum)

            # set weights to the last updated weights for the last minibatch

            weights = last_update

            next_neural_mem_state = next_neural_mem_state._replace(
                weights = weights,
                states = past_state,
            )

        next_neural_mem_state = next_neural_mem_state._replace(updates = updates)

        # retrieve

        if is_single_token:
            last_update, _ = next_neural_mem_state.states
            updates = rearrange_dict_values(last_update, 'b ... -> b 1 ...')

        retrieved = self.retrieve_memories(
            retrieve_seq,
            updates
        )

        # maybe detach

        if detach_mem_state:
            next_neural_mem_state = mem_state_detach(next_neural_mem_state)

        # returning

        if not return_surprises:
            return retrieved, next_neural_mem_state

        return retrieved, next_neural_mem_state, surprises
