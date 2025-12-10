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

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

class Transformer(Module):
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
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        for _ in range(depth):

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

            ff = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, int(dim * ff_expansion)),
                nn.GELU(),
                nn.Linear(int(dim * ff_expansion), dim)
            )

            self.layers.append(ModuleList([attn, ff]))

        self.norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9,
    ):
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        for _ in range(sample_num_times):
            logits = self.forward(out, return_loss = False)
            logits = logits[:, -1]

            logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(logits, temperature = temperature, dim = -1)

            out = torch.cat((out, sample), dim = -1)

        return out[..., prompt_seq_len:]

    def forward(self, x, return_loss = False):

        if return_loss:
            x, target = x[:, :-1], x[:, 1:]

        seq_len, device = x.shape[-1], x.device

        tokens = self.token_emb(x)

        for attn, ff in self.layers:
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens

        embed = self.norm(tokens)
        logits = self.to_logits(embed)

        if not return_loss:
            return logits

        return F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            target
        )

model = Transformer(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    implicit_mlp_attn_hiddens = (64, 96, 64),
    use_nested_attn = True # test implicit mlp attn vs nested attn
)

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

# accelerate

accelerator = Accelerator()

model, optim, train_loader, val_loader = accelerator.prepare(model, optim, train_loader, val_loader)

# cycle

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        loss = model(data, return_loss = True)

        accelerator.backward(loss / GRAD_ACCUM_EVERY)

    accelerator.print(f"training loss: {loss.item():.3f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss = model(valid_data, return_loss = True)
            accelerator.print(f"validation loss: {loss.item():.3f}")

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = next(val_loader)[0, :PRIME_LENGTH]

        prime = decode_tokens(inp)
        accelerator.print(f"\n\n[prompt]: {prime}")

        prompt = inp[None, ...]

        sampled = model.sample(prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        accelerator.print(f"\n[generated]: {base_decode_output}\n\n")
