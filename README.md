# pytorch-llm-learning
# 🚀 PyTorch & LLM 全链路学习路线

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-🤗-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

> 本仓库记录了我从 **PyTorch 基础** 到 **大模型（LLM）训练、微调、推理与部署** 的完整学习过程。  
> 目标是具备独立完成 **数据 → 模型 → 训练 → 推理 → 部署** 的能力，并将其应用到实际项目与工作中。

---

## 📅 学习路线（优化版）

| 模块 | 内容 | 任务目标 | 成果展示 |
|------|------|---------|----------|
| 🥇 **模块 1** | PyTorch 基础 & 复现性 | 张量运算、Autograd、随机种子、GPU 切换 | [笔记](notes/01_pytorch_basics.md) |
| 🥈 **模块 2** | 神经网络搭建 + 数据管线 | `nn.Module`、`Dataset/DataLoader`、评估指标、早停 | [代码](code/02_nn_dataloader) |
| 📝 **模块 3** | Transformer 文本分类 | Self-Attention、Hugging Face Transformers | [IMDb 分类结果](outputs/module3_imdb.png) |
| ⚡ **模块 4** | 训练优化与稳定性 | AMP、梯度裁剪、梯度累积、`torch.compile` | [性能对比图](outputs/module4_amp_vs_fp32.png) |
| 💻 **模块 5** | 多卡与分布式 | `torchrun`、Accelerate、NCCL 配置 | [吞吐量对比](outputs/module5_ddp.png) |
| 🪶 **模块 6** | 参数高效微调 + 量化 | LoRA、bitsandbytes 8/4bit | [推理延迟对比](outputs/module6_quant.png) |
| 🌐 **模块 7** | 推理与部署闭环 | 推理参数调优、Gradio/FastAPI 部署 | [Demo截图](outputs/module7_gradio.png) |
| 🎯 **模块 8** | 综合小项目 | RAG 问答系统 / Agent 应用 / 行业案例 | [项目演示](outputs/module8_demo.gif) |

---

## 📂 仓库结构
```
pytorch-llm-learning/
│
├── notes/            # 学习笔记（Markdown）
├── code/             # 每个模块的代码
├── outputs/          # 实验结果、截图、日志
├── README.md         # 本文件
└── requirements.txt  # 依赖环境
```

---

## 🔧 环境配置
```bash
conda create -n llm-env python=3.10
conda activate llm-env

pip install torch torchvision torchaudio
pip install transformers datasets accelerate peft bitsandbytes
pip install tensorboard matplotlib scikit-learn
```

---

## 📊 可视化（TensorBoard）
![TensorBoard 示例](outputs/sample_tensorboard.png)
```bash
tensorboard --logdir=./outputs/tensorboard_logs --port=6006
```
> 从 **模块 2** 开始，全程记录 loss、accuracy、学习率、梯度范数等。

---

## 🏆 成果亮点（持续更新）
- ✅ 从零实现线性回归（PyTorch）
- ✅ MNIST 分类准确率 ≥ 97%
- ✅ IMDb 文本分类 F1 ≥ 0.9
- ✅ AMP + LoRA 微调 GPT2，显存占用降低 60%
- ⏳ 综合小项目：RAG 问答系统 Demo

---

## 📌 学习笔记更新计划
- 每完成一个模块 → 笔记更新到 `notes/`  
- 代码与运行结果同步到 `code/` 与 `outputs/`  
- 遇到问题记录到 `notes/debug_log.md`（方便复盘）

---

模块 1｜PyTorch 基础与复现性
张量创建与运算（CPU/GPU 切换）
Autograd & 反向传播
torch.no_grad() 与 torch.inference_mode()
固定随机种子、打印环境信息
模型模式切换：train() / eval()
Debug 手段：print(tensor.shape), torch.cuda.memory_allocated()
任务：
波士顿房价（或合成数据）线性回归
固定种子复现实验结果

模块 2｜神经网络搭建 + 数据管线
自定义 nn.Module
Dataset & DataLoader
损失函数与优化器（MSELoss、CrossEntropyLoss）
学习率调度器（StepLR, CosineAnnealingLR）
提前停止（EarlyStopping）
任务：
MNIST/Fashion-MNIST 分类（acc ≥ 97%）
保存最佳权重 & 完整 checkpoint

模块 3｜Transformer 最小可用（文本分类）
Self-Attention 直观理解
Hugging Face transformers + datasets
AutoTokenizer & AutoModelForSequenceClassification
指标计算（accuracy、F1、混淆矩阵）
任务：
IMDb / AG News 分类
输出测试集指标 & 保存 tokenizer + 模型
模块 4｜训练优化与稳定性
AMP（autocast, GradScaler）
梯度裁剪（clip_grad_norm_）
梯度累积（gradient accumulation）
PyTorch 2.x torch.compile 加速
NaN 检测 & 日志监控
任务：
在模块3模型上对比：AMP vs FP16 vs FP32（显存、速度、指标变化）

模块 5｜多卡与分布式
torchrun --nproc_per_node=...
Hugging Face Accelerate (accelerate config / accelerate launch)
NCCL 后端 & 环境检查
梯度同步机制
任务：
2 卡并行跑 IMDb/BERT 微调
吞吐量对比单卡
扩展：
DeepSpeed ZeRO Stage 2/3（可选）

模块 6｜参数高效微调（PEFT/LoRA）+ 量化
LoRA 原理与 Hugging Face PEFT 库
bitsandbytes 8bit/4bit 量化
LoRA + AMP + 梯度累积结合
任务：
在 GPT-2 / Tiny-LLM 上做指令/对话微调
对比 FP16 与 8bit 推理延迟与显存占用
模块 7｜推理与部署闭环
推理参数（top-k, top-p, temperature）
保存模型/tokenizer/config
Gradio / FastAPI 简易部署
ONNX/TensorRT 导出（可选）
任务：
做一个网页对话 Demo
显示推理时间、显存占用
对比量化前后效果

模块 8｜综合小项目（选题灵活）

建议选题：
RAG 问答系统：用 sentence-transformers 检索课程笔记 → GPT-2 LoRA 回答问题

行业应用：如情感分析+客服对话，或数据分析报告生成