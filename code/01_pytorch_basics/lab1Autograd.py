# 实验目标
# 构造一个最小的两层网络（手动设置参数）
# 使用 autograd 自动求梯度
# 打印每一步的结果和梯度
import torch
import torch.nn as nn
import torch.nn.functional as F

# 固定随机种子，确保复现
torch.manual_seed(42)

# ===== 1. 构造两层网络 =====
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)  # 输入 3 维 → 隐藏层 4 维
        self.fc2 = nn.Linear(4, 1)  # 隐藏层 4 维 → 输出 1 维

    def forward(self, x):
        x = self.fc1(x)          # 第一层
        x = torch.relu(x)        # 激活
        x = self.fc2(x)          # 第二层
        return x

model = TwoLayerNet()

# ===== 2. 输入和标签 =====
x = torch.randn(2, 3)  # batch_size=2, 特征数=3
y_true = torch.tensor([[1.0], [0.0]])  # 目标输出

print("输入 x:\n", x)
print("真实标签 y_true:\n", y_true)

# ===== 3. 前向传播 =====
y_pred = model(x)
print("\n预测 y_pred:\n", y_pred)

# ===== 4. 损失计算 =====
loss = F.mse_loss(y_pred, y_true)
print("\n损失 Loss:", loss.item())

# ===== 5. 查看反向传播前的梯度（全是 None） =====
print("\n反向传播前参数梯度：")
for name, param in model.named_parameters():
    print(name, param.grad)

# ===== 6. 反向传播 =====
loss.backward()

# ===== 7. 查看反向传播后的梯度 =====
print("\n反向传播后参数梯度：")
for name, param in model.named_parameters():
    print(name, param.grad)

# ===== 8. 手动更新（模拟优化器） =====
learning_rate = 0.1
with torch.no_grad():
    for param in model.parameters():
        param -= learning_rate * param.grad

print("\n参数已更新。")
#运行后你会看到：
# 输入数据和标签

# 预测结果

# 损失值

# 反向传播前梯度是 None（还没计算）

# 反向传播后每个参数的梯度值（fc1.weight, fc1.bias, fc2.weight, fc2.bias）

# 手动更新参数后模型就完成了一步训练

# 💡 你可以多运行几次，会发现：

# 如果不固定随机种子，梯度和结果会变。

# 如果固定种子，梯度每次都一样（这就是复现性的重要性）。

# 改学习率，更新幅度也会变。