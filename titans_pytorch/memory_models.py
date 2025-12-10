"""
各种内存模型实现

这个模块包含了Titans论文中使用的多种神经网络内存组件实现，
包括不同结构的内存MLP和内存注意力模型。这些模型用于构建
具有长序列建模能力的Transformer架构。
"""
import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList, ParameterDict

from einops import rearrange  # 用于张量维度重排的工具

# 辅助函数

def l2norm(t):
    """对张量进行L2归一化"""
    return F.normalize(t, dim = -1)

# 归一化层

class LayerNorm(Module):
    """自定义层归一化
    
    与标准LayerNorm不同，这个实现使用了一个可学习的缩放参数gamma，
    并支持批量维度的gamma参数。
    """
    def __init__(
        self,
        dim  # 输入特征维度
    ):
        super().__init__()

        # 基础层归一化，不使用可学习的仿射参数
        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        # 可学习的缩放参数
        self.gamma = Parameter(torch.zeros(dim))

    def forward(self, x):
        """前向传播"""
        gamma = self.gamma

        # 如果gamma有批量维度，调整其形状以匹配输入
        if gamma.ndim == 2:
            gamma = rearrange(gamma, 'b d -> b 1 d')

        # 应用归一化并进行缩放
        return self.ln(x) * (gamma + 1.)

# 残差归一化包装器
class ResidualNorm(Module):
    """残差归一化包装器
    
    这是原始TTT论文中使用的包装器，将模型输出归一化后与输入相加。
    可以根据需要移除这个包装器。
    """
    def __init__(
        self,
        dim,  # 输入/输出特征维度
        model: Module  # 要包装的模型
    ):
        super().__init__()
        self.norm = LayerNorm(dim)  # 归一化层
        self.model = model  # 被包装的模型

    def forward(self, x):
        """前向传播"""
        # 模型计算
        out = self.model(x)
        # 归一化后残差连接
        return self.norm(out) + x

# 内存MLP模型
class MemoryMLP(Module):
    """内存MLP模型
    
    这是TTT论文中提出的基本内存MLP实现。
    它是一个简单的前馈神经网络，具有固定的深度和扩张因子。
    """
    def __init__(
        self,
        dim,  # 输入/输出特征维度
        depth,  # 模型深度（层数）
        expansion_factor = 2.  # 隐藏层的扩张因子
    ):
        super().__init__()
        # 计算隐藏层维度
        dim_hidden = int(dim * expansion_factor)
        # 构建各层的维度列表
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        # 创建权重参数列表
        self.weights = ParameterList([Parameter(torch.randn(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        # 使用Xavier均匀分布初始化权重
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(
        self,
        x  # 输入张量，形状为 (..., dim)
    ):
        """前向传播"""
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0  # 检查是否是第一层

            # 第一层之后应用GELU激活函数
            if not is_first:
                x = F.gelu(x)

            # 矩阵乘法
            x = x @ weight

        return x

# 带门控残差的内存MLP模型
class GatedResidualMemoryMLP(Module):
    """带门控残差的内存MLP模型
    
    这是一个改进的内存MLP模型，它使用门控残差连接和最终投影层。
    门控机制允许模型学习在原始输入和分支输出之间进行插值。
    """
    def __init__(
        self,
        dim,  # 输入/输出特征维度
        depth,  # 模型深度
        expansion_factor = 4.  # 隐藏层的扩张因子
    ):
        super().__init__()
        # 计算隐藏层维度
        dim_hidden = int(dim * expansion_factor)

        # 创建权重参数列表，每层包含三个权重矩阵
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, dim_hidden)),  # 第一个线性层权重
                Parameter(torch.randn(dim_hidden, dim)),  # 第二个线性层权重
                Parameter(torch.randn(dim * 2, dim)),     # 门控投影权重
            ]) for _ in range(depth)
        ])

        # 最终投影层
        self.final_proj = Parameter(torch.randn(dim, dim))

        # 使用Xavier均匀分布初始化所有参数
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(
        self,
        x  # 输入张量，形状为 (..., dim)
    ):
        """前向传播"""
        # 遍历所有层
        for weight1, weight2, to_gates in self.weights:
            res = x  # 保存残差连接

            # 分支计算
            hidden = x @ weight1
            hidden = F.gelu(hidden)
            branch_out = hidden @ weight2

            # 门控残差连接
            gates = cat((branch_out, res), dim = -1) @ to_gates  # 计算门控
            x = res.lerp(branch_out, gates.sigmoid())  # 在残差和分支输出之间插值

        # 最终投影
        return x @ self.final_proj

# 因子化权重的内存MLP模型
class FactorizedMemoryMLP(Module):
    """因子化权重的内存MLP模型
    
    这个模型使用因子化的权重矩阵，将原始的权重分解为两个较小的矩阵。
    这样可以在保持模型容量的同时，减少参数量和计算复杂度，
    特别适合于较小的块大小。
    """
    def __init__(
        self,
        dim,  # 输入/输出特征维度
        depth,  # 模型深度
        k = 32  # 因子化的中间维度
    ):
        super().__init__()
        # 创建因子化权重参数列表
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, k)),  # 第一个因子矩阵
                Parameter(torch.randn(k, dim)),  # 第二个因子矩阵
            ]) for _ in range(depth)
        ])

        # 使用Xavier均匀分布初始化所有权重
        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)

    def forward(
        self,
        x  # 输入张量，形状为 (..., dim)
    ):
        """前向传播"""
        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0  # 检查是否是第一层

            # 第一层之后应用GELU激活函数
            if not is_first:
                x = F.gelu(x)

            # 因子化矩阵乘法
            x = x @ weight1 @ weight2

        return x

# 使用SwiGLU激活的内存MLP模型
class MemorySwiGluMLP(Module):
    """使用SwiGLU激活的内存MLP模型
    
    这个模型模仿了现代Transformer中流行的SwiGLU前馈网络结构。
    它使用SwiGLU激活函数，这是一种门控线性单元变体，
    通常比标准GELU在Transformer架构中表现更好。
    """
    def __init__(
        self,
        dim,  # 输入/输出特征维度
        depth = 1,  # 模型深度，每个深度对应两个线性层
        expansion_factor = 4.  # 隐藏层的扩张因子
    ):
        super().__init__()

        # 计算内部维度，SwiGLU通常使用 2/3 * expansion_factor * dim
        dim_inner = int(dim * expansion_factor * 2 / 3)

        # 创建权重列表
        weights = []

        # 为每个深度创建两个线性层
        for _ in range(depth):
            weights.append(ParameterList([
                Parameter(torch.randn(dim, dim_inner * 2)),  # 第一个线性层，输出维度是dim_inner * 2
                Parameter(torch.randn(dim_inner, dim)),     # 第二个线性层
            ]))

        self.weights = ParameterList(weights)
        self.norm = LayerNorm(dim)  # 层归一化

    def forward(self, x):
        """前向传播"""
        for w1, w2 in self.weights:
            residual = x  # 保存残差连接

            # SwiGLU激活
            x, gates = (x @ w1).chunk(2, dim = -1)  # 将第一个线性层的输出分为两部分
            x = x * F.gelu(gates)  # 应用门控

            # 第二个线性层
            x = x @ w2

            # 残差连接
            x = x + residual

        return x

# 内存注意力模块

class MemoryAttention(Module):
    """内存注意力模型
    
    这是一个改进的注意力机制实现，将注意力和前馈网络并行计算，
    类似于PaLM和Gpt-J中的设计。它使用固定权重矩阵实现注意力，
    可以作为内存模块使用。
    
    Args:
        dim: 输入/输出特征维度
        scale: 注意力缩放因子
        expansion_factor: 前馈网络隐藏层的扩张因子
    """
    def __init__(
        self,
        dim,
        scale = 8.,
        expansion_factor = 2.
    ):
        super().__init__()
        self.scale = scale  # 注意力缩放因子
        dim_ff_hidden = int(dim * expansion_factor)  # 前馈网络隐藏层维度

        # 创建权重参数列表
        self.weights = ParameterList([
            Parameter(torch.randn(dim, dim)), # 查询投影权重
            Parameter(torch.randn(dim, dim)), # 键投影权重
            Parameter(torch.randn(dim, dim)), # 值投影权重
            Parameter(torch.randn(dim, dim_ff_hidden)), # 前馈网络第一层权重
            Parameter(torch.randn(dim_ff_hidden, dim)), # 前馈网络第二层权重
        ])

        # 使用Xavier均匀分布初始化所有权重
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (..., dim)
            
        Returns:
            输出张量，形状与输入相同
        """
        # 解包权重参数
        wq, wk, wv, ffw1, ffw2 = self.weights

        # 生成查询、键、值
        q = l2norm(x @ wq)  # 生成查询并归一化
        k = l2norm(x @ wk)  # 生成键并归一化
        v = x @ wv  # 生成值

        # 计算注意力输出
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,  # 使用自定义缩放因子
            is_causal = True  # 使用因果掩码（防止未来信息泄露）
        )

        # 并行计算前馈网络输出
        # 类似于PaLM和Gpt-J的设计
        h = F.gelu(x @ ffw1)  # 前馈网络第一层，应用GELU激活
        ff_out = h @ ffw2  # 前馈网络第二层

        # 返回注意力和前馈网络的输出之和
        return attn_out + ff_out
