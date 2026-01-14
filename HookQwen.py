from transformers.generation.utils import GenerateOutput
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

##防御干预的逻辑实现
class InterventionalSAEProbe:
    def __init__(self, sae, threshold=0.2, steering_coeff=1.5):
        self.sae = sae
        self.threshold = threshold
        self.steering_coeff = steering_coeff # 转向强度
        self.target_feature_idx = 42        # 假设 42 号特征代表“专业性”
        self.log_data = ["test"]
        
    def hook_fn(self, module, input, output):
        # 1. 提取原始激活值
        is_tuple = isinstance(output, tuple)
        original_acts = output[0] if is_tuple else output
        
        # 2. SAE 映射：x -> f -> x_hat
        with torch.no_grad():
            # 编码
            hidden_pre = self.sae.encoder(original_acts) + self.sae.encoder_bias
            feature_acts = torch.relu(hidden_pre)
            
            # --- 核心干预逻辑 ---
            # 场景 A: 增强特定特征 (Feature Steering)
            feature_acts[..., self.target_feature_idx] *= self.steering_coeff
            
            # 场景 B: 异常抑制 (Out-of-distribution Defense)
            reconstructed_x = self.sae.decoder(feature_acts)
            mse = torch.norm(original_acts - reconstructed_x, dim=-1)
            
            if mse.mean() > self.threshold:
                # 如果误差过大，说明进入了不安全或不确定区域
                # 我们可以通过融合重构值来“纠偏”，或者直接减弱激活
                modified_acts = 0.7 * original_acts + 0.3 * reconstructed_x
            else:
                modified_acts = reconstructed_x # 使用经过 SAE 过滤后的纯净特征
        
        # 3. 将修改后的激活值写回模型
        if is_tuple:
            return (modified_acts,) + output[1:]
        return modified_acts

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
    ## 这里可以修改为InterventionalSAEProbe
    sae = SparseAutoencoder(d_model, dict_size).to(torch.float16)

    # 实例化探针
    probe = SAEProbe(sae, threshold=0.1)

    # 寻找接入点：Qwen 的结构通常是 layers[i].mlp
    # 我们把探针挂载到第 2 层 MLP 上
    target_layer = model.model.layers[2].mlp
    handle = target_layer.register_forward_hook(probe.hook_fn)

    # ==========模拟推理=============
    prompt = "2026.1.14北京的天气是什么？"
  
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