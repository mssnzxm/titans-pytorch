# /// script
# dependencies = [
#     "accelerate",      # 用于分布式训练加速
#     "titans-pytorch",  # 包含自定义的注意力机制实现
#     "tqdm"             # 进度条显示
# ]
# ///

import math
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam      # Adam优化器
from torch import nn, Tensor      # 神经网络模块和张量
from torch.nn import Module, ModuleList  # 模块基类和模块列表
import torch.nn.functional as F  # 神经网络功能函数
from torch.utils.data import DataLoader, Dataset  # 数据加载器和数据集

from einops import rearrange      # 张量维度重排工具

from titans_pytorch.implicit_mlp_attention import ImplicitMLPAttention  # 隐式MLP注意力机制
from titans_pytorch.nested_attention import NestedAttention            # 嵌套注意力机制

from accelerate import Accelerator  # 分布式训练加速器

# 常量定义

NUM_BATCHES = int(1e5)      # 训练批次数
BATCH_SIZE = 4              # 批处理大小
GRAD_ACCUM_EVERY = 4        # 梯度累积步数
LEARNING_RATE = 1e-4        # 学习率
VALIDATE_EVERY = 100        # 每100个批次验证一次
PRIME_LENGTH = 32           # 生成文本时的提示长度
GENERATE_EVERY = 250        # 每250个批次生成一次文本
GENERATE_LENGTH = 512       # 生成文本的长度
SEQ_LEN = 512               # 序列长度

# 辅助函数

def exists(v):
    """检查值是否存在（不为None）"""
    return v is not None

def cycle(loader):
    """循环迭代数据加载器"""
    while True:
        for data in loader:
            yield data

def decode_token(token):
    """将token解码为字符，确保ASCII值不小于32（可打印字符）"""
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    """将token序列解码为字符串"""
    return "".join(list(map(decode_token, tokens)))

# 采样辅助函数

def log(t, eps = 1e-20):
    """安全的对数函数，避免数值溢出"""
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    """生成Gumbel噪声"""
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    """使用Gumbel噪声进行采样"""
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    """对logits应用top-k过滤，只保留概率最高的k个token"""
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

class Transformer(Module):
    """Transformer模型，使用ImplicitMLPAttention或NestedAttention"""
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        implicit_mlp_attn_hiddens = (64, 96, 64),
        use_nested_attn = False,
        dim_head = 64,
        ff_expansion = 4.,
        attn_kwargs: dict = dict(),
    ):
        """
        参数:
        - num_tokens: 词汇表大小
        - dim: 模型维度
        - depth: 模型深度
        - heads: 注意力头数
        - implicit_mlp_attn_hiddens: 隐式MLP注意力的隐藏层维度
        - use_nested_attn: 是否使用嵌套注意力机制
        - dim_head: 每个注意力头的维度
        - ff_expansion: 前馈网络的扩展因子
        - attn_kwargs: 注意力机制的额外参数
        """
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)  # token嵌入层

        self.layers = ModuleList([])  # 存储模型层

        for _ in range(depth):
            # 根据配置选择注意力机制
            if use_nested_attn:
                attn = NestedAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    **attn_kwargs
                )
            else:
                attn = ImplicitMLPAttention(
                    dim = dim,
                    mlp_hiddens = implicit_mlp_attn_hiddens,
                    heads = heads,
                    **attn_kwargs
                )

            # 前馈网络
            ff = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, int(dim * ff_expansion)),
                nn.GELU(),
                nn.Linear(int(dim * ff_expansion), dim)
            )

            self.layers.append(ModuleList([attn, ff]))

        self.norm = nn.RMSNorm(dim)  # 最终的归一化层
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)  # 输出层，生成logits

    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9,
    ):
        """
        根据提示生成文本
        
        参数:
        - prompt: 提示文本的token序列
        - seq_len: 生成的文本总长度
        - temperature: 采样温度
        - filter_thres: top-k过滤阈值
        
        返回值:
        - 生成的token序列
        """
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        for _ in range(sample_num_times):
            logits = self.forward(out, return_loss = False)  # 获取logits
            logits = logits[:, -1]  # 只取最后一个token的logits

            logits = top_k(logits, thres = filter_thres)  # top-k过滤
            sample = gumbel_sample(logits, temperature = temperature, dim = -1)  # 采样

            out = torch.cat((out, sample), dim = -1)  # 将采样结果添加到输出序列

        return out[..., prompt_seq_len:]  # 返回新生成的部分

    def forward(self, x, return_loss = False):
        """
        模型前向传播
        
        参数:
        - x: 输入token序列
        - return_loss: 是否返回损失
        
        返回值:
        - logits或损失
        """
        if return_loss:
            x, target = x[:, :-1], x[:, 1:]  # 准备输入和目标序列

        seq_len, device = x.shape[-1], x.device

        tokens = self.token_emb(x)  # token嵌入

        # 通过所有层
        for attn, ff in self.layers:
            tokens = attn(tokens) + tokens  # 注意力层 + 残差连接
            tokens = ff(tokens) + tokens  # 前馈网络 + 残差连接

        embed = self.norm(tokens)  # 最终归一化
        logits = self.to_logits(embed)  # 生成logits

        if not return_loss:
            return logits

        # 计算损失
        return F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            target
        )

# 初始化模型
model = Transformer(
    num_tokens = 256,  # 词汇表大小为256（ASCII字符）
    dim = 512,         # 模型维度
    depth = 6,         # 6层
    implicit_mlp_attn_hiddens = (64, 96, 64),  # 隐式MLP注意力的隐藏层
    use_nested_attn = True  # 使用NestedAttention，设置为False可使用ImplicitMLPAttention
)

# 准备enwik8数据

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()  # 读取并转换数据
    np_train, np_valid = np.split(data, [int(90e6)])  # 分割训练集和验证集
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)  # 转换为PyTorch张量

class TextSamplerDataset(Dataset):
    """文本采样数据集，用于生成训练样本"""
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data      # 原始数据
        self.seq_len = seq_len  # 序列长度

    def __len__(self):
        """返回数据集大小"""
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        """获取一个训练样本"""
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))  # 随机起始位置
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()  # 提取序列
        return full_seq

# 创建数据集和数据加载器
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# 初始化优化器
optim = Adam(model.parameters(), lr = LEARNING_RATE)

# 初始化Accelerator
executor = Accelerator()

# 准备模型、优化器和数据加载器
model, optim, train_loader, val_loader = executor.prepare(model, optim, train_loader, val_loader)

# 循环迭代数据加载器
train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# 训练循环

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()  # 设置模型为训练模式

    # 梯度累积
    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)  # 获取下一个批次的数据

        loss = model(data, return_loss = True)  # 计算损失

        executor.backward(loss / GRAD_ACCUM_EVERY)  # 反向传播，注意损失要除以梯度累积步数

    executor.print(f"training loss: {loss.item():.3f}")  # 打印训练损失

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪

    optim.step()  # 更新参数
    optim.zero_grad()  # 清空梯度

    # 验证
    if i % VALIDATE_EVERY == 0:
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            valid_data = next(val_loader)  # 获取验证数据

            loss = model(valid_data, return_loss = True)  # 计算验证损失
            executor.print(f"validation loss: {loss.item():.3f}")  # 打印验证损失

    # 生成文本
    if i % GENERATE_EVERY == 0:
        model.eval()  # 设置模型为评估模式

        inp = next(val_loader)[0, :PRIME_LENGTH]  # 获取提示文本

        prime = decode_tokens(inp)  # 解码提示文本
        executor.print(f"\n\n[prompt]: {prime}")  # 打印提示文本

        prompt = inp[None, ...]  # 添加批次维度

        sampled = model.sample(prompt, GENERATE_LENGTH)  # 生成文本

        base_decode_output = decode_tokens(sampled[0])  # 解码生成的文本

        executor.print(f"\n[generated]: {base_decode_output}\n\n")  # 打印生成的文本
