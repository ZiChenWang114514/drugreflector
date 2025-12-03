# HCA-DR: Hierarchical Cell-Aware DrugReflector

基于细胞系感知的药物反应预测模型，使用FiLM调制实现细胞系特异性建模。

## 📁 项目结构

```
hca_dr/
├── __init__.py      # 包初始化
├── config.py        # 配置定义
├── model.py         # 模型架构
├── dataset.py       # 数据集定义
├── losses.py        # 损失函数
├── trainer.py       # 训练器（三阶段训练）
├── train.py         # 训练脚本
└── eval.py          # 评估脚本
```

## 🏗️ 模型架构

HCA-DR包含四个核心模块：

1. **全局扰动编码器 (Global Perturbation Encoder)**
   - 输入：978维扰动签名
   - 架构：978 → 1024 → 2048
   - 输出：全局特征 h_global

2. **细胞系上下文编码器 (Cell Context Encoder)**
   - 输入：978维细胞系上下文（INT归一化）
   - 架构：978 → 256
   - 输出：上下文特征 h_ctx

3. **自适应FiLM调制 (Adaptive FiLM Modulation)**
   - 生成 γ, β 进行特征调制
   - 自适应混合权重 α 控制回退
   - 输出：调制特征 h_adapted = (1-α)·h_global + α·(γ⊙h_global + β)

4. **分类头 (Classification Head)**
   - 输入：2048维特征
   - 输出：药物类别logits

## 📊 三阶段训练策略

### 阶段1：全局模型预训练 (Epoch 1-20)
- **目标**：复现DrugReflector，建立强大的全局基线
- **训练**：全局编码器 + 分类头
- **冻结**：上下文编码器 + FiLM模块
- **损失**：Focal Loss

### 阶段2：FiLM分支训练 (Epoch 21-40)
- **目标**：让FiLM学习全局模型的残差
- **训练**：上下文编码器 + FiLM模块 + 分类头
- **冻结**：全局编码器
- **损失**：Focal Loss + Contrastive Loss + α-Penalty

### 阶段3：端到端微调 (Epoch 41-50)
- **目标**：联合优化所有参数
- **训练**：所有参数
- **损失**：完整损失函数（权重调整）

## 🔧 损失函数

```
L_total = L_drug + λ₁·L_contrast + λ₂·L_global + λ₃·L_α

其中：
- L_drug: Focal Loss (γ=2)
- L_contrast: Supervised Contrastive Loss (τ=0.1)
- L_global: 全局模型Focal Loss
- L_α: Alpha Penalty (Context Dropout时惩罚非零α)
```

## 🚀 快速开始

### 1. 数据预处理

首先运行数据预处理脚本生成HCA-DR训练数据：

```bash
python hca_dr_data_preprocessing.py
```

### 2. 训练模型

```bash
python train.py \
    --data_path /path/to/hca_dr_training_data.pkl \
    --output_dir /path/to/outputs \
    --batch_size 256 \
    --stage1_epochs 20 \
    --stage2_epochs 20 \
    --stage3_epochs 10
```

### 3. 评估模型

```bash
python eval.py \
    --checkpoint /path/to/checkpoint.pt \
    --data_path /path/to/hca_dr_training_data.pkl \
    --output_dir /path/to/results
```

## 📈 评估指标

1. **Top-k% Recall**: 药物被预测在top-k%的比例
2. **LOCO (Leave-One-Cell-Out)**: 在未见细胞系上的泛化能力
3. **OOD Evaluation**: 零向量上下文时的回退能力
4. **Alpha Analysis**: α值分布分析

## 🎯 预期结果

| 指标 | DrugReflector | HCA-DR (预期) |
|------|---------------|---------------|
| CMap Top 1% Recall | 0.81 | 0.82-0.84 |
| LOCO Recall | 0.65 | 0.70-0.75 |
| OOD Recall | 0.48 | 0.55-0.60 |
| α_seen | N/A | 0.75-0.85 |
| α_OOD | N/A | 0.05-0.15 |

## 📦 依赖

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
tqdm>=4.62.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 📝 配置参数

主要超参数：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| batch_size | 256 | 批大小 |
| stage1_lr | 0.0139 | 阶段1学习率 |
| stage2_lr | 0.01 | 阶段2学习率 |
| stage3_lr | 0.001 | 阶段3学习率 |
| encoder_dropout | 0.64 | 编码器dropout率 |
| context_dropout | 0.15 | Context Dropout概率 |
| focal_gamma | 2.0 | Focal Loss gamma |
| contrast_temp | 0.1 | 对比学习温度 |
| lambda_contrast | 0.1 | 对比损失权重 |
| lambda_global | 0.3 | 全局正则化权重 |
| lambda_alpha | 0.5 | Alpha Penalty权重 |

## 📄 License

MIT License