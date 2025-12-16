
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple
import math

class TitansMemoryModule(nn.Module):
    """Titans记忆模块"""
    def __init__(self, hidden_size: int, memory_slots: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.memory = nn.Parameter(torch.randn(memory_slots, hidden_size))
        self.read_head = nn.Linear(hidden_size, hidden_size)
        self.write_head = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        # 读取记忆
        read_weights = torch.softmax(torch.matmul(x, self.memory.transpose(0, 1)), dim=-1)
        read_memory = torch.matmul(read_weights, self.memory)
        # 写入记忆
        write_content = self.write_head(x.mean(dim=1))
        self.memory.data = 0.9 * self.memory.data + 0.1 * write_content.unsqueeze(1)
        # 返回读取的记忆和原始输入的结合
        combined = torch.cat([x, read_memory], dim=-1)
        return combined, read_memory

class QwenWithTitans(nn.Module):
    """集成Titans记忆模块的千问模型"""
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B", memory_slots: int = 64):
        super().__init__()
        # 加载预训练千问模型
        self.qwen_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 获取千问模型的隐藏层大小
        self.hidden_size = self.qwen_model.config.hidden_size
        
        # 初始化Titans记忆模块
        self.memory_module = TitansMemoryModule(self.hidden_size, memory_slots)
        
        # 适配层，用于调整维度
        self.adapter = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 获取千问模型的隐藏状态
        with torch.no_grad():
            outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # 获取最后一层隐藏状态
            
        # 应用Titans记忆模块
        enhanced_hidden, memory = self.memory_module(hidden_states)
        
        # 通过适配层调整维度
        adapted_hidden = self.adapter(enhanced_hidden)
        
        # 使用增强后的隐藏状态生成输出
        logits = self.qwen_model.lm_head(adapted_hidden)
        
        return logits, memory

class TitansQwenProcessor:
    """处理千问-Titans集成模型的输入输出"""
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B", memory_slots: int = 64):
        self.model = QwenWithTitans(model_name, memory_slots)
        self.tokenizer = self.model.tokenizer
        
    def generate_with_memory(self, prompt: str, max_length: int = 100, temperature: float = 0.7):
        """使用集成模型生成文本"""
        # 编码输入
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        generated_tokens = []
        current_input_ids = input_ids
        
        # 逐个生成token
        for _ in range(max_length):
            # 前向传播
            with torch.no_grad():
                logits, _ = self.model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask
                )
            
            # 应用温度参数
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # 采样下一个token
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            generated_tokens.append(next_token.item())
            
            # 更新输入
            current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((attention_mask.shape[0], 1), dtype=torch.long)
            ], dim=1)
            
            # 检查是否生成结束符
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
    
    def get_memory_state(self):
        """获取当前记忆状态"""
        return self.model.memory_module.memory.detach().cpu().numpy()

def main():
    """主函数演示Titans与千问模型集成"""
    print("正在初始化千问-Titans集成模型...")
    
    try:
        # 初始化集成模型处理器
        processor = TitansQwenProcessor("Qwen/Qwen2-0.5B", memory_slots=32)
        print("模型初始化成功！")
        
        # 测试文本生成
        prompt = "人工智能的未来发展"
        print(f"\n生成提示: {prompt}")
        
        generated_text = processor.generate_with_memory(
            prompt=prompt,
            max_length=50,
            temperature=0.8
        )
        
        print(f"生成结果: {generated_text}")
        
        # 显示记忆状态
        memory_state = processor.get_memory_state()
        print(f"\n当前记忆状态形状: {memory_state.shape}")
        print("记忆模块已激活，模型在生成过程中使用了上下文记忆机制")
        
        # 第二次生成以展示记忆效果
        print("\n=== 第二次生成以展示记忆效果 ===")
        prompt2 = "基于前面讨论的人工智能发展趋势"
        generated_text2 = processor.generate_with_memory(
            prompt=prompt2,
            max_length=50,
            temperature=0.8
        )
        print(f"第二次生成结果: {generated_text2}")
        
    except Exception as e:
        print(f"模型初始化失败: {e}")
        print("请确保已安装所需依赖: pip install transformers torch")

if __name__ == "__main__":
    main()
