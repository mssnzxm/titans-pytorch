import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,  # 记忆上下文Transformer
    MemoryMLP,  # 记忆MLP
    MemoryAttention  # 记忆注意力
)

# 模块说明：
# 此文件用于训练具有记忆功能的上下文Transformer模型(MemoryAsContextTransformer)
# 使用enwik8数据集进行文本生成任务，演示了如何结合神经记忆机制增强Transformer的长期依赖能力

# 常量定义

NUM_BATCHES = int(1e5)  # 训练批次数
BATCH_SIZE = 4  # 批次大小
GRADIENT_ACCUMULATE_EVERY = 4  # 梯度累积步数
LEARNING_RATE = 2e-4  # 学习率
VALIDATE_EVERY  = 100  # 验证间隔
GENERATE_EVERY  = 500  # 生成文本间隔
PRIME_LENGTH = 100  # 提示文本长度
GENERATE_LENGTH = 512  # 生成文本长度
SHOULD_GENERATE = True  # 是否生成文本
SEQ_LEN = 512  # 序列长度

# 神经记忆相关参数

NEURAL_MEMORY_DEPTH = 2  # 神经记忆深度
NUM_PERSIST_MEM = 4  # 持久记忆数量
NUM_LONGTERM_MEM = 4  # 长期记忆数量
NEURAL_MEM_LAYERS = (2, 4, 6)  # 具有神经记忆的层索引
NEURAL_MEM_GATE_ATTN_OUTPUT = False  # 是否使用门控注意力输出
NEURAL_MEM_MOMENTUM = True  # 是否启用动量
NEURAL_MEM_MOMENTUM_ORDER = 1  # 动量阶数
NEURAL_MEM_QK_NORM = True  # 是否对查询和键进行归一化
NEURAL_MEM_MAX_LR = 1e-1  # 神经记忆最大学习率
USE_MEM_ATTENTION_MODEL = False  # 是否使用记忆注意力模型
WINDOW_SIZE = 32  # 窗口大小
NEURAL_MEM_SEGMENT_LEN = 4  # 神经记忆段长度（设置越小，学习率/动量等的粒度越细）
NEURAL_MEM_BATCH_SIZE = 128  # 神经记忆批次大小（设置越小，在遍历序列时更新神经记忆权重的频率越高）
SLIDING_WINDOWS = True  # 是否使用滑动窗口
STORE_ATTN_POOL_CHUNKS = True  # 是否使用注意力池化处理块导出的动量、每层学习率调整和衰减
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True  # 是否为每层学习率建模
NEURAL_MEM_WEIGHT_RESIDUAL = True  # 学习接受来自前一个神经记忆层权重的贡献会带来显著改进
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True  # 允许神经记忆选择从哪些层派生查询/键/值
NEURAL_MEM_SPEC_NORM_SURPRISES = True  # 通过对惊喜进行谱归一化，将Muon优化器的经验应用于惊喜更新

# 实验相关参数

PROJECT_NAME = 'titans-mac-transformer'  # 项目名称
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'  # 运行名称
WANDB_ONLINE = False  # 是否在线使用wandb（设置为True可将实验数据上传到云端）

# 性能相关参数

USE_ACCELERATED_SCAN = True  # 是否使用加速扫描
USE_FLEX_ATTN = True  # 是否使用灵活注意力
USE_FAST_INFERENCE = False  # 是否使用快速推理

# wandb experiment tracker

import wandb
wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# 辅助函数

def cycle(loader):
    """循环迭代数据加载器"""
    while True:
        for data in loader:
            yield data

def decode_token(token):
    """解码单个令牌为字符"""
    return str(chr(max(32, token)))  # 确保ASCII码不小于32（可打印字符）

def decode_tokens(tokens):
    """解码令牌序列为字符串"""
    return ''.join(list(map(decode_token, tokens)))  # 合并解码后的字符

# 记忆模型定义

if USE_MEM_ATTENTION_MODEL:
    # 使用记忆注意力模型
    neural_memory_model = MemoryAttention(
        dim = 64  # 维度
    )
else:
    # 使用记忆MLP模型
    neural_memory_model = MemoryMLP(
        dim = 64,  # 维度
        depth = NEURAL_MEMORY_DEPTH  # 深度
    )

# 实例化记忆上下文Transformer模型

model = MemoryAsContextTransformer(
    num_tokens = 256,  # 令牌数量
    dim = 384,  # 隐藏层维度
    depth = 8,  # 深度
    segment_len = WINDOW_SIZE,  # 段长度
    num_persist_mem_tokens = NUM_PERSIST_MEM,  # 持久记忆令牌数量
    num_longterm_mem_tokens = NUM_LONGTERM_MEM,  # 长期记忆令牌数量
    neural_memory_layers = NEURAL_MEM_LAYERS,  # 具有神经记忆的层
    neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,  # 神经记忆段长度
    neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,  # 神经记忆批次大小
    neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,  # 是否使用门控注意力输出
    neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,  # 是否使用权重残差
    neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,  # 是否为QKV提供不同视图
    use_flex_attn = USE_FLEX_ATTN,  # 是否使用灵活注意力
    sliding_window_attn = SLIDING_WINDOWS,  # 是否使用滑动窗口注意力
    neural_memory_model = neural_memory_model,  # 神经记忆模型
    neural_memory_kwargs = dict(  # 神经记忆参数
        dim_head = 64,  # 头维度
        heads = 4,  # 注意力头数量
        attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,  # 是否使用注意力池化块
        qk_rmsnorm = NEURAL_MEM_QK_NORM,  # 是否对QK进行RMS归一化
        momentum = NEURAL_MEM_MOMENTUM,  # 是否启用动量
        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,  # 动量阶数
        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,  # 默认步变换最大学习率
        use_accelerated_scan = USE_ACCELERATED_SCAN,  # 是否使用加速扫描
        per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR,  # 是否对每个参数进行学习率调制
        spectral_norm_surprises = NEURAL_MEM_SPEC_NORM_SURPRISES  # 是否对惊喜进行谱归一化
    )
).cuda()  # 将模型移动到GPU

# 准备enwik8数据

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()  # 读取并转换数据
    data_train, data_val = np.split(data, [int(90e6)])  # 分割训练集和验证集
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))  # 转换为PyTorch张量

class TextSamplerDataset(Dataset):
    """文本采样数据集，用于生成训练样本"""
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data  # 原始数据
        self.seq_len = seq_len  # 序列长度

    def __getitem__(self, index):
        """获取一个训练样本"""
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))  # 随机起始位置
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()  # 提取序列
        return full_seq.cuda()  # 将数据移动到GPU

    def __len__(self):
        """返回数据集大小"""
        return self.data.size(0) // self.seq_len

# 创建数据集和数据加载器
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# 初始化优化器

optim = AdoptAtan2(model.parameters(), lr = LEARNING_RATE)  # 使用AdoptAtan2优化器

# 训练循环

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    model.train()  # 设置模型为训练模式

    # 梯度累积
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)  # 计算损失
        loss.backward()  # 反向传播

    print(f'training loss: {loss.item()}')  # 打印训练损失
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
    optim.step()  # 更新参数
    optim.zero_grad()  # 清空梯度
    wandb.log(dict(loss = loss.item()))  # 记录损失到wandb

    # 验证
    if i % VALIDATE_EVERY == 0:
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)  # 计算验证损失
            print(f'validation loss: {loss.item()}')  # 打印验证损失

    # 生成文本
    if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
        model.eval()  # 设置模型为评估模式
        inp = random.choice(val_dataset)[:PRIME_LENGTH]  # 随机选择提示文本
        prime = decode_tokens(inp)  # 解码提示文本
        print(f'%s \n\n %s', (prime, '*' * 100))  # 打印提示文本和分隔符

        sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = USE_FAST_INFERENCE)  # 生成文本
        output_str = decode_tokens(sample[0])  # 解码生成的文本
        print(output_str)  # 打印生成的文本
