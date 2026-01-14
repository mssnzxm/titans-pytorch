from transformers.generation.utils import GenerateOutput
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 定义简化的 SAE 模块 (Sparse Autoencoder)
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        
        # 编码器：将激活值投影到高维稀疏空间
        self.encoder = nn.Linear(d_model, dict_size)
        self.encoder_bias = nn.Parameter(torch.zeros(dict_size))
        
        # 解码器：尝试重构原始激活值
        self.decoder = nn.Linear(dict_size, d_model)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # SAE 运算核心逻辑
        hidden_pre = self.encoder(x) + self.encoder_bias
        feature_acts = torch.relu(hidden_pre)  # 稀疏特征 f
        
        reconstructed_x = self.decoder(feature_acts)
        
        # 计算 L2 重构误差 (用于衡量“不确定性”或“异常”)
        reconstruction_error = torch.norm(x - reconstructed_x, dim=-1)
        
        return feature_acts, reconstruction_error

# 2. 定义探针管理器 (基于 PyTorch Hook)
class SAEProbe:
    def __init__(self, sae_model, threshold=0.5):
        self.sae = sae_model
        self.threshold = threshold
        self.log_data = []

    def hook_fn(self, module, input, output):
        # output 通常是 (batch, seq, hidden_size) 的 Tensor
        # 如果 output 是 tuple (在某些模型实现中)，取第一项
        activations = output[0] if isinstance(output, tuple) else output
        
        # 执行 SAE 运算
        with torch.no_grad():
            features, error = self.sae(activations)
        
        # 防御干预逻辑：检测重构误差是否超过阈值（代表模型进入了 SAE 未训练过的分布）
        mean_error = error.mean().item()
        if mean_error > self.threshold:
            print(f"[警告] 探针检测到高不确定性! Error: {mean_error:.4f}")
            # 这里可以执行干预，例如修改 output 或记录日志
            
        self.log_data.append({
            "sparsity": (features > 0).float().mean().item(),
            "error": mean_error
        })

# 3. 主程序：加载 Qwen3 并注入探针
def run_sae_probe_demo():
    model_name =  "./Qwen3-0.6B"  # 假设使用 Qwen 家族模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="cpu"
    )

    # 配置 SAE：假设 Qwen 0.5B 的 hidden_size 是 1024
    d_model = model.config.hidden_size
    dict_size = d_model * 4  # 常见的 SAE 扩展倍数
    sae = SparseAutoencoder(d_model, dict_size).to(torch.float16)

    # 实例化探针
    probe = SAEProbe(sae, threshold=0.1)

    # 寻找接入点：Qwen 的结构通常是 layers[i].mlp
    # 我们把探针挂载到第 2 层 MLP 上
    target_layer = model.model.layers[2].mlp
    handle = target_layer.register_forward_hook(probe.hook_fn)

    # ==========模拟推理=============
    prompt = "北京今天的天气是什么？"
  
    enable_thinking: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

     # 应用对话模板
     # 构建对话消息
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
     )
        
     # 准备模型输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("开始推理...")
    with torch.no_grad():
            generated_ids: GenerateOutput | torch.LongTensor = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )
     # 提取生成的token
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
            # 查找<|thinking|>标记
        index = len(output_ids) - output_ids[::-1].index(151668)  # 151668是<|thinking|>的token_id
    except ValueError:
        index = 0
        
        # 解码思考内容和回答内容
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print(f"提示: {prompt}")
    print(f"思考内容: {thinking_content}")
    print(f"回答: {content}")

    
    print("\n推理完成。")
    print(f"探针日志示例: {probe.log_data[-1]}")

    # 记得移除 Hook 避免内存泄漏
    handle.remove()

if __name__ == "__main__":
    run_sae_probe_demo()