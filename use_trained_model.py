# /// script
# dependencies = [
#     "torch",           # PyTorch框架
#     "titans-pytorch",  # 包含自定义的注意力机制实现
#     "einops"           # 张量维度重排工具
# ]
# ///

import torch
import os
import sys
import argparse
from torch import nn, Tensor
from einops import rearrange

# 从train_implicit_mlp_attn.py中导入必要的组件
from train_implicit_mlp_attn import Transformer, decode_token, decode_tokens

# 定义采样辅助函数
def log(t, eps=1e-20):
    """安全的对数函数，避免数值溢出"""
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    """生成Gumbel噪声"""
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1, keepdim=True):
    """使用Gumbel噪声进行采样"""
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim, keepdim=keepdim)

def top_k(logits, thres=0.9):
    """对logits应用top-k过滤，只保留概率最高的k个token"""
    k = int(math.ceil((1 - thres) * logits.shape[-1]))
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def load_model(model_path, device="cpu"):
    """
    加载训练好的模型
    
    参数:
    - model_path: 模型文件路径
    - device: 运行设备（"cpu"或"cuda"）
    
    返回值:
    - 加载好的模型
    """
    # 模型配置参数，需要与训练时保持一致
    model_config = {
        "num_tokens": 256,
        "dim": 512,
        "depth": 6,
        "implicit_mlp_attn_hiddens": (64, 96, 64),
        "use_nested_attn": True
    }
    
    # 创建模型实例
    model = Transformer(**model_config)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载模型状态字典
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 设置模型为评估模式
    model.eval()
    
    # 移动模型到指定设备
    model.to(device)
    
    return model

def generate_text(model, prompt, seq_len=512, temperature=1.0, filter_thres=0.9, device="cpu"):
    """
    使用模型生成文本
    
    参数:
    - model: 加载好的模型
    - prompt: 提示文本
    - seq_len: 生成的文本总长度
    - temperature: 采样温度
    - filter_thres: top-k过滤阈值
    - device: 运行设备（"cpu"或"cuda"）
    
    返回值:
    - 生成的文本
    """
    # 将提示文本转换为token序列
    prompt_tokens = torch.tensor([ord(c) for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
    
    # 生成文本
    with torch.no_grad():
        generated_tokens = model.sample(prompt_tokens, seq_len, temperature, filter_thres)
    
    # 将生成的token序列解码为文本
    generated_text = decode_tokens(generated_tokens.squeeze().tolist())
    
    return generated_text

def main():
    """主函数"""
    print("\n=== 使用训练好的模型生成文本 ===")
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用训练好的模型生成文本")
    parser.add_argument("--model_path", type=str, default="./models/best_model.pt", 
                      help="训练好的模型路径")
    parser.add_argument("--prompt", type=str, default="Hello, world!", 
                      help="生成文本的提示")
    parser.add_argument("--seq_len", type=int, default=200, 
                      help="生成的文本长度")
    parser.add_argument("--temperature", type=float, default=1.0, 
                      help="采样温度")
    parser.add_argument("--filter_thres", type=float, default=0.9, 
                      help="top-k过滤阈值")
    parser.add_argument("--device", type=str, default="cpu", 
                      help="运行设备（cpu或cuda）")
    
    args = parser.parse_args()
    
    # 检查设备是否可用
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = "cpu"
    
    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    try:
        model = load_model(args.model_path, args.device)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)
    
    # 生成文本
    print(f"\n提示文本: {args.prompt}")
    print(f"生成文本长度: {args.seq_len}")
    print(f"采样温度: {args.temperature}")
    print(f"Top-k过滤阈值: {args.filter_thres}")
    print("\n生成中...")
    
    generated_text = generate_text(
        model,
        args.prompt,
        args.seq_len,
        args.temperature,
        args.filter_thres,
        args.device
    )
    
    # 打印生成的文本
    print("\n=== 生成的文本 ===")
    print(args.prompt + generated_text)
    print("==================")

if __name__ == "__main__":
    # 导入math模块（在函数内部使用时需要）
    import math
    main()