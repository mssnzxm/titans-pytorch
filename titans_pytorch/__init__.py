# 从 neural_memory 模块导入核心神经网络内存组件
from titans_pytorch.neural_memory import (
    NeuralMemory,  # 神经网络内存主类，用于在测试时学习记忆
    NeuralMemState,  # 神经网络内存状态数据结构
    mem_state_detach  # 将内存状态从计算图中分离的工具函数
)

# 从 memory_models 模块导入各种内存模型实现
from titans_pytorch.memory_models import (
    MemoryMLP,  # 基本的内存MLP模型
    MemoryAttention,  # 基于注意力的内存模型
    FactorizedMemoryMLP,  # 因子化的内存MLP模型，用于更高效的内存使用
    MemorySwiGluMLP,  # 使用SwiGLU激活的内存MLP模型
    GatedResidualMemoryMLP  # 带门控残差连接的内存MLP模型
)

# 从 mac_transformer 模块导入基于内存上下文的Transformer
from titans_pytorch.mac_transformer import (
    MemoryAsContextTransformer  # 使用内存作为上下文的Transformer架构，实现Titans模型
)
