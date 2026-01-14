import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from HookQwen import SparseAutoencoder

class SAETrainer:
    def __init__(self, sae, lr=1e-3, l1_coeff=0.01):
        self.sae = sae
        self.l1_coeff = l1_coeff  # 控制稀疏性的超参数
        self.optimizer = optim.Adam(sae.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, batch_acts):
        """
        batch_acts: [batch_size, d_model] - 从原模型收集的激活值
        """
        self.optimizer.zero_grad()
        
        # 1. 前向传播
        # feature_acts: 稀疏特征 (f)
        # reconstructed_x: 重构后的激活值 (x_hat)
        hidden_pre = self.sae.encoder(batch_acts) + self.sae.encoder_bias
        feature_acts = torch.relu(hidden_pre)
        reconstructed_x = self.sae.decoder(feature_acts)
        
        # 2. 计算损失函数
        # MSE Loss: 保证重构质量
        mse_loss = self.criterion(reconstructed_x, batch_acts)
        
        # L1 Loss: 强制特征稀疏化 (鼓励大部分神经元为 0)
        l1_loss = feature_acts.abs().sum(dim=-1).mean()
        
        total_loss = mse_loss + self.l1_coeff * l1_loss
        
        # 3. 反向传播
        total_loss.backward()
        
        # 梯度裁剪：防止训练崩溃（可选，但在训练 SAE 时很常用）
        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
        
        self.optimizer.step()
        
        return total_loss.item(), mse_loss.item(), l1_loss.item()

# --- 模拟训练流程 ---

def train_demo():
    # 参数配置
    d_model = 1024   # Qwen-0.5B 的维度
    dict_size = d_model * 4
    batch_size = 32
    
    # 初始化 SAE
    sae = SparseAutoencoder(d_model, dict_size)
    trainer = SAETrainer(sae, l1_coeff=5e-4)

    # 模拟从 Qwen 模型中收集到的激活值数据
    # 实际操作中，你需要先运行前一节的 Hook 代码，把激活值存入 list，然后转换成 Tensor
    dummy_activations = torch.randn(1000, d_model) 
    dataset = TensorDataset(dummy_activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("开始训练 SAE 探针...")
    for epoch in range(5):
        for i, (batch,) in enumerate(loader):
            loss, mse, l1 = trainer.train_step(batch)
            
            if i % 10 == 0:
                # 计算当前的稀疏度 (L0 norm)
                with torch.no_grad():
                    f = torch.relu(sae.encoder(batch) + sae.encoder_bias)
                    l0 = (f > 0).float().sum(dim=-1).mean().item()
                
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss:.4f} | MSE: {mse:.4f} | L0 (活跃特征数): {l0:.1f}")

    # 保存训练好的探针权重
    torch.save(sae.state_dict(), "qwen_sae_probe.pt")
    print("训练完成，探针权重已保存。")

if __name__ == "__main__":
    train_demo()