from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Qwen3_0_6B:
    """调用本地Qwen3-0.6B模型的类"""
    def __init__(self, model_path: str = "./Qwen3-0.6B", device: str = None):
        """
        初始化模型和分词器
        
        Args:
            model_path: 本地模型路径
            device: 运行设备，如"cuda:0"或"cpu"，默认为自动检测
        """
        # 自动检测设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"正在加载模型到设备: {device}...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device,
            trust_remote_code=True
        )
        
        print("模型加载完成！")
    
    def generate_response(self, prompt: str, enable_thinking: bool = True, 
                          max_new_tokens: int = 1024, temperature: float = 0.6,
                          top_p: float = 0.95, top_k: int = 20):
        """
        生成模型响应
        
        Args:
            prompt: 用户输入的提示
            enable_thinking: 是否启用思考模式
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: top-p采样参数
            top_k: top-k采样参数
            
        Returns:
            tuple: (思考内容, 回答内容)
        """
        # 构建对话消息
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 应用对话模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # 准备模型输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 生成文本
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )
        
        # 提取生成的token
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # 解析思考内容和回答内容
        try:
            # 查找</think>标记
            index = len(output_ids) - output_ids[::-1].index(151668)  # 151668是</think>的token_id
        except ValueError:
            index = 0
        
        # 解码思考内容和回答内容
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return thinking_content, content
    
    def multi_turn_conversation(self, max_turns: int = 10):
        """
        多轮对话交互
        
        Args:
            max_turns: 最大对话轮数
        """
        print("=== Qwen3-0.6B 多轮对话 ===")
        print("输入 'exit' 或 '退出' 结束对话")
        print("输入 'toggle' 或 '切换' 切换思考模式")
        print("当前思考模式: 启用")
        
        enable_thinking = True
        conversation_history = []
        
        for turn in range(max_turns):
            user_input = input(f"\n用户 (轮次 {turn+1}/{max_turns}): ")
            
            if user_input.lower() in ["exit", "退出"]:
                print("对话结束")
                break
            
            if user_input.lower() in ["toggle", "切换"]:
                enable_thinking = not enable_thinking
                print(f"思考模式已{'启用' if enable_thinking else '禁用'}")
                continue
            
            # 添加到对话历史
            conversation_history.append({"role": "user", "content": user_input})
            
            # 应用对话模板
            text = self.tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            
            # 准备模型输入
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            print(f"Qwen3-0.6B {'(思考模式)' if enable_thinking else '(快速模式)'}: ", end="")
            
            # 生成文本
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    do_sample=True
                )
            
            # 提取生成的token
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # 解析思考内容和回答内容
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                if thinking_content and enable_thinking:
                    print(f"\n思考过程: {thinking_content}\n回答: ", end="")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            except ValueError:
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
            print(content)
            
            # 添加到对话历史
            conversation_history.append({"role": "assistant", "content": content})
        
        print(f"\n最大对话轮数 ({max_turns}) 已达到，对话结束")

def main():
    """主函数，演示模型使用"""
    try:
        # 初始化模型
        qwen3 = Qwen3_0_6B()
        
        # 单轮对话示例
        print("\n=== 单轮对话示例 ===")
        prompt = "请简要介绍一下大语言模型"
        thinking, response = qwen3.generate_response(prompt, enable_thinking=True)
        print(f"提示: {prompt}")
        print(f"思考内容: {thinking}")
        print(f"回答: {response}")
        
        # 快速模式示例
        print("\n=== 快速模式示例 ===")
        prompt = "2+2等于多少"
        thinking, response = qwen3.generate_response(prompt, enable_thinking=False)
        print(f"提示: {prompt}")
        print(f"回答: {response}")
        
        # 启动多轮对话
        qwen3.multi_turn_conversation()
        
    except Exception as e:
        print(f"模型调用失败: {e}")
        print("请确保已安装所需依赖: pip install transformers torch")

if __name__ == "__main__":
    main()