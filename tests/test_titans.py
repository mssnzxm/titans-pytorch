from contextlib import contextmanager

import torch
from torch import nn

import pytest
from titans_pytorch import NeuralMemory
from titans_pytorch.mac_transformer import flex_attention, SegmentedAttention, MemoryAsContextTransformer

# 辅助函数

def exists(v):
    """检查值是否存在"""
    return v is not None

def diff(x, y):
    """计算两个张量之间的最大绝对差异"""
    return (x - y).abs().amax()

@contextmanager
def torch_default_dtype(dtype):
    """临时设置PyTorch默认数据类型的上下文管理器"""
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# 主测试函数

@pytest.mark.parametrize('seq_len', (32, 512, 77))
@pytest.mark.parametrize('silu', (False, True))
@pytest.mark.parametrize('chunk_size, attn_pool_chunks', ((64, True), (64, False), (1, False)))
@pytest.mark.parametrize('momentum', (False, True))
@pytest.mark.parametrize('qk_rmsnorm', (False, True))
@pytest.mark.parametrize('heads', (1, 4))
@pytest.mark.parametrize('max_grad_norm', (None, 2.))
@pytest.mark.parametrize('num_kv_per_token', (1, 2))
@pytest.mark.parametrize('per_parameter_lr_modulation', (False, True))
@pytest.mark.parametrize('per_head_learned_parameters', (False, True))
@pytest.mark.parametrize('test_store_mask', (False, True))
def test_titans(
    seq_len,
    silu,
    attn_pool_chunks,
    chunk_size,
    momentum,
    qk_rmsnorm,
    heads,
    max_grad_norm,
    num_kv_per_token,
    per_parameter_lr_modulation,
    per_head_learned_parameters,
    test_store_mask
):
    """测试NeuralMemory的基本功能"""
    # 创建NeuralMemory实例
    mem = NeuralMemory(
        dim = 16,
        chunk_size = chunk_size,
        activation = nn.SiLU() if silu else None,
        attn_pool_chunks = attn_pool_chunks,
        max_grad_norm = max_grad_norm,
        num_kv_per_token = num_kv_per_token,
        momentum = momentum,
        qk_rmsnorm = qk_rmsnorm,
        heads = heads,
        per_parameter_lr_modulation = per_parameter_lr_modulation,
        per_head_learned_parameters = per_head_learned_parameters
    )

    # 创建随机输入序列
    seq = torch.randn(2, seq_len, 16)

    # 准备存储掩码
    store_mask = None
    if test_store_mask:
        store_mask = torch.randint(0, 2, (2, seq_len)).bool()

    # 测试NeuralMemory的前向传播
    retrieved, _ = mem(seq, store_mask = store_mask)

    # 验证输出形状与输入相同
    assert seq.shape == retrieved.shape

def test_return_surprises():
    """测试返回惊喜值和自适应学习率"""
    mem = NeuralMemory(
        dim = 384,
        chunk_size = 2,
        dim_head = 64,
        heads = 4,
    )

    seq = torch.randn(4, 64, 384)

    # 测试返回惊喜值和自适应学习率
    _, _, (surprises, adaptive_lr) = mem(seq, return_surprises = True)

    # 验证惊喜值和自适应学习率的形状
    assert all([t.shape == (4, 4, 64) for t in (surprises, adaptive_lr)])

@pytest.mark.parametrize('learned_momentum_combine', (False, True))
@pytest.mark.parametrize('learned_combine_include_zeroth', (False, True))
def test_titans_second_order_momentum(
    learned_momentum_combine,
    learned_combine_include_zeroth
):
    """测试二阶动量"""
    mem  = NeuralMemory(
        dim = 384,
        dim_head = 64,
        heads = 2,
        chunk_size = 1,
        batch_size = 2,
        momentum_order = 2,  # 设置动量阶数为2
        learned_momentum_combine = learned_momentum_combine,
        learned_combine_include_zeroth = learned_combine_include_zeroth
    )

    seq = torch.randn(2, 5, 384)

    # 测试二阶动量的前向传播
    parallel_retrieved, state = mem(seq)
    assert seq.shape == parallel_retrieved.shape

def test_titans_attn_memory():
    """测试使用MemoryAttention作为记忆模型的NeuralMemory"""
    from titans_pytorch.memory_models import MemoryAttention

    mem = NeuralMemory(
        dim = 16,
        chunk_size = 64,
        model = MemoryAttention(
            dim = 16
        )
    )

    seq = torch.randn(2, 1024, 16)
    retrieved, _ = mem(seq)

    assert seq.shape == retrieved.shape

def test_swiglu_ff_memory():
    """测试使用MemorySwiGluMLP作为记忆模型的NeuralMemory"""
    from titans_pytorch.memory_models import MemorySwiGluMLP

    mem = NeuralMemory(
        dim = 16,
        chunk_size = 2,
        mem_model_norm_add_residual = False,
        model = MemorySwiGluMLP(
            dim = 16,
            depth = 2
        )
    )

    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)

    assert seq.shape == retrieved.shape

@pytest.mark.parametrize('gated_transition', (True, False))
def test_neural_mem_chaining_chunks(
    gated_transition
):
    """测试NeuralMemory的分块链式处理"""
    mem  = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 16,
        gated_transition = gated_transition
    )

    seq = torch.randn(2, 48, 16)

    # 测试并行处理
    parallel_retrieved, state = mem(seq)

    # 分割序列为多个块
    seq_first, seq_second, seq_third = seq.split(16, dim = 1)

    # 测试链式处理
    first_retrieved, state = mem(seq_first)
    second_retrieved, state = mem(seq_second, state = state)
    third_retrieved, state = mem(seq_third, state = state)

    # 验证并行处理和链式处理的结果一致
    assert torch.allclose(parallel_retrieved, torch.cat((first_retrieved, second_retrieved, third_retrieved), dim = 1), atol = 1e-5)

def test_neural_mem_chaining_with_weight_residual():
    """测试带有权重残差的NeuralMemory链式处理"""
    # 创建第一个NeuralMemory实例
    mem  = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 64
    )

    # 创建接受权重残差的第二个NeuralMemory实例
    mem2 = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 64,
        accept_weight_residual = True
    )

    seq = torch.randn(2, 256, 16)

    # 通过第一个NeuralMemory处理序列
    seq, state = mem(seq)

    # 测试并行处理（带权重残差）
    parallel_retrieved, _ = mem2(seq, prev_weights = state.updates)

    # 分割序列为两部分
    seq_first, seq_second = seq[:, :128], seq[:, 128:]

    # 测试链式处理（带权重残差）
    first_retrieved, state1 = mem2(seq_first, prev_weights = state.updates)
    second_retrieved, state2 = mem2(seq_second, state = state1, prev_weights = state.updates)

    # 验证并行处理和链式处理的结果一致
    assert torch.allclose(parallel_retrieved, torch.cat((first_retrieved, second_retrieved), dim = 1), atol = 1e-5)

def test_neural_mem_chaining_with_batch_size():
    """测试带有批量大小的NeuralMemory链式处理"""
    mem  = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 16,
        batch_size = 64  # 设置批量大小
    )

    seq = torch.randn(2, 112, 16)

    # 测试并行处理
    parallel_retrieved, state = mem(seq)

    # 分割序列为三个部分
    seq_first, seq_second, seq_third = seq[:, :16], seq[:, 16:64], seq[:, 64:]

    # 测试链式处理
    first_retrieved, state = mem(seq_first)
    second_retrieved, state = mem(seq_second, state = state)
    third_retrieved, state = mem(seq_third, state = state)

    parallel_part_retrieved = torch.cat((first_retrieved, second_retrieved, third_retrieved), dim = 1)

    # 验证并行处理和链式处理的结果一致
    assert torch.allclose(parallel_retrieved, parallel_part_retrieved, atol = 1e-5)

@pytest.mark.parametrize('seq_len', (1023, 17))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 16))
@pytest.mark.parametrize('num_longterm_mem_tokens', (0, 16))
@pytest.mark.parametrize('neural_mem_gate_attn_output', (False, True))
@pytest.mark.parametrize('neural_mem_segment_len', (8, 16))
@pytest.mark.parametrize('neural_mem_weight_residual', (False, True))
@pytest.mark.parametrize('neural_mem_batch_size', (None, 64))
@pytest.mark.parametrize('neural_mem_qkv_receives_diff_views', (False, True))
@pytest.mark.parametrize('neural_mem_momentum', (False, True))
def test_mac(
    seq_len,
    num_persist_mem_tokens,
    num_longterm_mem_tokens,
    neural_mem_gate_attn_output,
    neural_mem_segment_len,
    neural_mem_weight_residual,
    neural_mem_batch_size,
    neural_mem_qkv_receives_diff_views,
    neural_mem_momentum
):
    """测试MemoryAsContextTransformer的基本功能"""
    # 创建MemoryAsContextTransformer实例
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 16,
        depth = 2,
        num_persist_mem_tokens = num_persist_mem_tokens,
        num_longterm_mem_tokens = num_longterm_mem_tokens,
        segment_len = 128,
        neural_mem_gate_attn_output = neural_mem_gate_attn_output,
        neural_memory_segment_len = neural_mem_segment_len,
        neural_memory_batch_size = neural_mem_batch_size,
        neural_memory_qkv_receives_diff_views = neural_mem_qkv_receives_diff_views,
        neural_mem_weight_residual = neural_mem_weight_residual,
        neural_memory_kwargs = dict(
            momentum = neural_mem_momentum
        )
    )

    # 创建随机输入序列
    x = torch.randint(0, 256, (1, seq_len))

    # 测试前向传播
    logits = transformer(x)
    assert logits.shape == (1, seq_len, 256)

@pytest.mark.parametrize('sliding', (False, True))
@pytest.mark.parametrize('mem_layers', ((), None))
@pytest.mark.parametrize('longterm_mems', (0, 4, 16))
@pytest.mark.parametrize('prompt_len', (4, 16))
@torch_default_dtype(torch.float64)
def test_mac_sampling(
    sliding,
    mem_layers,
    longterm_mems,
    prompt_len
):
    """测试MemoryAsContextTransformer的采样功能"""
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 16,
        depth = 4,
        segment_len = 32,
        num_persist_mem_tokens = 4,
        num_longterm_mem_tokens = longterm_mems,
        sliding_window_attn = sliding,
        neural_memory_layers = mem_layers,
        neural_mem_gate_attn_output = False
    )

    ids = torch.randint(0, 256, (1, 1023))

    # 模拟训练后

    prompt = ids[:, :prompt_len]  # 提取提示文本

    # 测试不带缓存的采样
    sampled = transformer.sample(prompt, 53, use_cache = False, temperature = 0.)
    # 测试带缓存的采样
    sampled_with_cache = transformer.sample(prompt, 53, use_cache = True, temperature = 0.)

    # 验证两种采样方式的结果一致
    assert torch.allclose(sampled, sampled_with_cache)

@pytest.mark.parametrize('seq_len', (2, 64, 256))
@pytest.mark.parametrize('prompt_len', (0, 65))
@pytest.mark.parametrize('mem_chunk_size', (2, 32, 64))
@pytest.mark.parametrize('gated_transition', (False, True))
@torch_default_dtype(torch.float64)
def test_neural_mem_inference(
    seq_len,
    prompt_len,
    mem_chunk_size,
    gated_transition
):
    """测试NeuralMemory的推理功能"""
    mem = NeuralMemory(
        dim = 16,
        chunk_size = mem_chunk_size,
        gated_transition = gated_transition
    )

    seq = torch.randn(2, seq_len, 16)
    # 测试并行推理
    parallel_retrieved, _ = mem(seq)
    assert seq.shape == parallel_retrieved.shape

    # 测试顺序推理
    state = None
    sequential_retrieved = []

    # 测试初始并行提示
    test_parallel_prompt = prompt_len > 0 and prompt_len < seq_len

    if test_parallel_prompt:
        prompt, seq = seq[:, :prompt_len], seq[:, prompt_len:]
        retrieved_prompt, state = mem(prompt)
        sequential_retrieved.append(retrieved_prompt)

    # 顺序推理
    for token in seq.unbind(dim = 1):
        one_retrieved, state = mem.forward(
            token,
            state = state,
        )
        sequential_retrieved.append(one_retrieved)

    sequential_retrieved = torch.cat(sequential_retrieved, dim = -2)

    # 验证并行推理和顺序推理的结果一致
    assert torch.allclose(parallel_retrieved, sequential_retrieved, atol = 1e-6)

@pytest.mark.parametrize('seq_len', (1023, 17))
@pytest.mark.parametrize('sliding', (True, False))
def test_flex(
    seq_len,
    sliding
):
    """测试灵活注意力机制"""
    # 检查CUDA是否可用且flex_attention是否存在
    if not (torch.cuda.is_available() and exists(flex_attention)):
        pytest.skip()

    # 创建SegmentedAttention实例
    attn = SegmentedAttention(
        dim = 16,
        segment_len = 32,
        num_persist_mem_tokens = 1,
        num_longterm_mem_tokens = 1,
        use_flex_attn = True,
        sliding = sliding
    ).cuda()

    seq = torch.randn(1, seq_len, 16).cuda()

    # 测试灵活注意力
    out_flex, _ = attn(seq)
    # 测试非灵活注意力
    out_non_flex, _ = attn(seq, disable_flex_attn = True)

    # 验证两种注意力机制的结果一致
    assert torch.allclose(out_flex, out_non_flex, atol = 1e-5)

@pytest.mark.parametrize('use_accelerated', (True, False))
def test_assoc_scan(
    use_accelerated
):
    """测试关联扫描"""
    from titans_pytorch.neural_memory import AssocScan

    if use_accelerated and not torch.cuda.is_available():
        pytest.skip()

    # 创建AssocScan实例
    scan = AssocScan(use_accelerated = use_accelerated)

    seq_len = 128
    mid_point = seq_len // 2

    gates = torch.randn(2, seq_len, 16).sigmoid()
    inputs = torch.randn(2, seq_len, 16)

    if use_accelerated:
        gates = gates.cuda()
        inputs = inputs.cuda()

    # 测试完整序列的关联扫描
    output = scan(gates, inputs)

    # 分割序列为两部分
    gates1, gates2 = gates[:, :mid_point], gates[:, mid_point:]
    inputs1, inputs2 = inputs[:, :mid_point], inputs[:, mid_point:]

    # 测试第一部分的关联扫描
    first_half = scan(gates1, inputs1)

    # 测试第二部分的关联扫描（带前一部分的输出）
    second_half = scan(gates2, inputs2, prev = first_half[:, -1])
    assert second_half.shape == inputs2.shape

    # 验证完整序列和分割序列的结果一致
    assert torch.allclose(output[:, -1], second_half[:, -1], atol = 1e-5)

def test_mem_state_detach():
    """测试记忆状态分离"""
    from titans_pytorch.neural_memory import mem_state_detach

    mem = NeuralMemory(
        dim = 384,
        chunk_size = 2,
        qk_rmsnorm = True,
        dim_head = 64,
        heads = 4,
    )

    seq = torch.randn(4, 64, 384)

    state = None

    # 测试多次前向传播和反向传播
    for _ in range(2):
        parallel_retrieved, state = mem(seq, state = state)
        state = mem_state_detach(state)  # 分离记忆状态
        parallel_retrieved.sum().backward()  # 反向传播
