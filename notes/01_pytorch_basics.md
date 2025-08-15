1. 什么是 PyTorch？

PyTorch 是一个 深度学习框架，核心是 张量 (Tensor) 和 计算图 (Computation Graph)。
你可以把它理解成 NumPy + GPU + 自动求导：
NumPy：可以做矩阵运算、数学计算。
GPU：可以用显卡加速运算。
自动求导：可以自动计算梯度，帮我们更新模型参数。

💡 在深度学习中，几乎所有东西（输入、权重、输出）都是 张量。

2. 张量（Tensor）
2.1 张量是什么

张量 = 数字表格（可以是标量、向量、矩阵、高维数组）。

| 维度   | 名称            | 例子                | 张量形状                               |
| ---- | ------------- | ----------------- | ---------------------------------- |
| 0 维  | 标量 (scalar)   | 3.14              | `torch.Size([])`                   |
| 1 维  | 向量 (vector)   | \[1, 2, 3]        | `(3,)`                             |
| 2 维  | 矩阵 (matrix)   | \[\[1,2], \[3,4]] | `(2, 2)`                           |
| 3+ 维 | 高维张量 (tensor) | 图像批次数据            | `(batch, height, width, channels)` |

# 0维张量（标量）
a = torch.tensor(3.14)
# 1维张量（向量）
b = torch.tensor([1, 2, 3])
# 2维张量（矩阵）
c = torch.tensor([[1, 2], [3, 4]])

#也可与 NumPy 互转
import numpy as np
np_array = np.array([[1, 2], [3, 4]])
tensor_from_np = torch.from_numpy(np_array)  # 共享内存
back_to_np = tensor_from_np.numpy()

💡 注意：from_numpy 得到的张量与原 NumPy 数组共享内存，改一个另一个也会变。

2.2 张量运算
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.tensor([4, 5, 6], dtype=torch.float32)

# 加减乘除：逐元素运算
print(x + y)

# 矩阵乘法，矩阵运算
mat1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
mat2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(torch.matmul(mat1, mat2))  # 结果是 2x2 矩阵
print(mat1 @ mat2)               # @ 是 matmul 的简写

维度操作
t = torch.randn((2, 3))
print(t.shape)       # (2, 3)
print(t.T.shape)     # 转置 (3, 2)
print(t.view(3, 2))  # 改变形状

3. 设备切换（CPU ↔ GPU）

深度学习一般用 GPU 训练，因为速度快很多。
PyTorch 里可以用 .to(device) 把张量移到 GPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

x = torch.rand((2, 3)).to(device)
print(x.device)
⚠️ 注意：不同设备的张量不能直接运算：
a = torch.tensor([1, 2, 3])  # CPU
b = torch.tensor([4, 5, 6]).to("cuda")  # GPU
# a + b 会报错，因为它们在不同设备
必须先 .to() 到同一设备再运算。
小练习（CPU/GPU 切换）一个简单任务：
创建一个 (1000, 1000) 的随机张量。
把它移动到 GPU（如果有）。
计算它和自己的矩阵乘积（matmul）。
打印耗时。
这样你能直观看出 CPU 和 GPU 的差异。

1️⃣ Autograd 是什么？
Auto：自动
Grad：gradient（梯度）
➡ Autograd 就是 自动计算梯度 的系统。
在深度学习里，训练就是不断：
计算预测值（forward 前向传播）
计算损失（loss）
计算损失对参数的梯度（backward 反向传播）
用梯度更新参数（优化器 step）

💡 为什么需要 Autograd？
手动写公式（比如链式法则）很容易出错，尤其是神经网络非常深的时候。
Autograd 能自动构建 计算图（computation graph），在 .backward() 时沿着图反向计算梯度。

2️⃣ 计算图的概念
每次你用 requires_grad=True 的张量做运算，PyTorch 会记录：
操作（+、×、矩阵乘等）
输入张量
输出张量
这些信息连成一张 有向无环图（DAG），用来追踪数据如何一步步计算出来。
当你调用 .backward() 时，PyTorch 会沿着这张图反向传递梯度。
3️⃣ 反向传播例子

假设我们有：

𝑦=𝑥2+3𝑥+1
y=x2+3x+1
手动求导：𝑑𝑦𝑑𝑥=2𝑥+3
当 x=2 时，梯度应为 7。
PyTorch 实现：
import torch
# 创建需要梯度的标量
x = torch.tensor(2.0, requires_grad=True)
# 前向计算
y = x**2 + 3*x + 1
# 反向传播
y.backward()
# 输出梯度
print(x.grad)  # tensor(7.)
💡 注意：
.backward() 会计算 y 对 x 的梯度，并存到 x.grad 里。
如果你不加 requires_grad=True，PyTorch 不会追踪梯度。

4️⃣ 多变量求导
如果 
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x**2).sum()
y.backward()
print(x.grad)  # tensor([2., 4.])

5️⃣ 梯度累积现象
在 PyTorch 中，梯度是累积的：
x = torch.tensor(2.0, requires_grad=True)
y1 = x * 2
y1.backward()
print(x.grad)  # 2

y2 = x * 3
y2.backward()
print(x.grad)  # 2 + 3 = 5

💡 所以在训练循环里通常会：
optimizer.zero_grad()
先清空上一次的梯度。
1️⃣ 为什么要这样设计？
主要有三个原因：
① 支持 梯度累积训练（Gradient Accumulation）
在显存不足时，我们可能不能一次喂一个很大的 batch，就会把大 batch 拆成多个小 batch，分别做 forward() → backward()，再一起 optimizer.step()。
如果梯度不累积，每次 backward() 就会把之前的小 batch 梯度覆盖掉，这种分批累积就没法实现。
# 梯度累积示例
accum_steps = 4
optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    output = model(x)
    loss = criterion(output, y)
    loss.backward()           # 梯度累积
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad() # 再清零
② 方便多任务 / 多 loss 累加
有时我们会计算多个损失并多次调用 backward()，例如多任务学习（Multi-task Learning）。
累积的机制让多次 backward() 的梯度能够叠加起来，等价于一次性对多个 loss 求和再 backward()。
loss1.backward(retain_graph=True)
loss2.backward()  # loss2 梯度会加到 loss1 梯度上
③ 避免隐式数据丢失
如果 PyTorch 每次都自动清空梯度，一旦你忘记保存它，梯度就没了。
显式清零可以让用户在控制流程上更安全，防止梯度被覆盖。
2️⃣ 实际行为
假设：
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.])
y = x ** 2
y.backward()
print(x.grad)  # tensor([8.])  ← 累积
3️⃣ 如何避免累积？
如果我们只想要当前 batch 的梯度，必须手动清零：
optimizer.zero_grad()  # 常用
# 或
model.zero_grad()
# 或
for p in model.parameters():
    p.grad = None  # 更高效，不会分配新 tensor