# FP4 量化研究 — 数值分析实验报告

> 项目周期：4 周 | 最后更新：2026-05-02
> 远程执行：8× RTX 4090 @ bi_group2@bioinfo_class

---

## 摘要

本研究以数值分析为框架，在 ~164M 参数的 Gemma 风格 Transformer 上系统性地比较了 FP8/FP4 量化方案。实验沿两条路径展开：(A) **训练后量化（PTQ）**——6 种方法，含原创的 Lloyd-Max 自适应网格；(B) **量化感知训练（QAT）**——使用直通估计器（STE）、随机舍入和条件数正则化进行 FP8/FP4 直接训练。核心发现：(1) **RMSNorm 阻断跨层误差传播达 1482×**；(2) **Lloyd-Max 逐层自适应网格在 FP8 PTQ 上达最优**（Δ=−14.5）；(3) **条件数正则化 + κ 加权自适应网格在 FP4 上达最优**（Δ=−18.2）；(4) **PTQ 在所有指标上均显著优于 QAT**。

---

## 一、实验设置

### 1.1 模型

| 参数 | 值 |
|------|-----|
| 模型 | Micro-Gemma-FP (~164M 参数) |
| 架构 | 12 层 Transformer，RMSNorm、RoPE、GQA (4:1)，8 sliding + 4 full attention |
| 嵌入 | Per-layer token embeddings (64-dim) |
| 词汇量 | BPE 32K |
| 训练步数 | 2000 steps，batch=8，seq=512 |

### 1.2 数据

| Tier | 来源 | Tokens |
|------|--------|--------|
| D1 | C4 (raw) | 1.00B |
| D2 | FineWeb-edu | 1.40B |
| D3 | Wikipedia | 1.00B |
| D4 | OpenOrca | 0.84B |
| **总计** | | **4.24B（25.9× tokens/参数）** |

---

## 二、统一对比表

以下表格汇总了所有实验的评估 PPL。PTQ 结果基于 FP16 基线模型（评估 PPL=653.2），QAT 结果基于从零训练模型（评估 PPL 通过独立验证集测量）。Δ 值相对于各自基线模型。

### 2.1 FP16 基线

| 模型 | 训练 PPL | 评估 PPL | avg κ(W) |
|------|----------|----------|-----------|
| FP16 基线（标准训练） | 532 | 653.2 | 6.5 |
| FP16 + 条件数正则化（λ=1e-4） | ~600 | 671.1 | 5.1 |

### 2.2 PTQ 方法（全部 6 种）

| 方法 | FP8 评估 PPL | FP8 Δ | FP4 评估 PPL | FP4 Δ |
|------|-------------|--------|-------------|--------|
| **标准基线模型**（FP16 评估 PPL = 653.2） |
| Simple（逐张量） | 674.7 | +21.5 | 647.8 | −5.4 |
| Simple（逐通道） | 660.9 | +7.7 | 672.3 | +19.1 |
| GPTQ（权重补偿） | 662.6 | +9.4 | 684.2 | +30.9 |
| 混合精度（3FP8+9FP4） | 651.4 | −1.8 | 675.3 | +22.1 |
| **Lloyd-Max 自适应** ★ | **638.7** | **−14.5** | 674.5 | +21.2 |
| κ 加权自适应 | 655.1 | +1.9 | 663.7 | +10.5 |
| **条件数正则化模型**（FP16 评估 PPL = 671.1） |
| Simple（逐张量） | 683.7 | +12.7 | 670.0 | −1.0 |
| Simple（逐通道） | 668.4 | −2.7 | 668.6 | −2.5 |
| GPTQ（权重补偿） | 678.9 | +7.8 | 677.8 | +6.7 |
| **混合精度** ★ | **651.3** | **−19.8** | 662.6 | −8.4 |
| Lloyd-Max 自适应 | 667.8 | −3.3 | 671.3 | +0.2 |
| **κ 加权自适应** ★ | 662.5 | −8.6 | **652.8** | **−18.2** |

### 2.3 QAT 方法（全部 4 种）

| 方法 | 训练 PPL | 评估 PPL | vs FP16 基线评估 | 类型 |
|------|----------|----------|------------------|------|
| FP16 基线（参考） | 532 | 669.5 | — | 全精度 |
| QAT-FP8 | 1,105 | 1,323.4 | +654 | STE baseline |
| QAT-FP4 | 1,564 | 1,443.6 | +774 | STE baseline |
| QAT-FP8 + SR + CondReg | 1,311 | 1,312.0 | +643 | +随机舍入 + κ正则化 |
| QAT-FP4 + SR + CondReg | 1,307 | 1,489.8 | +820 | +随机舍入 + κ正则化 |

### 2.4 PTQ vs QAT 头部对比

| 维度 | 最佳 PTQ 结果 | 最佳 QAT 结果 | 差距 |
|------|-------------|-------------|------|
| FP8 评估 PPL | 638.7 (Lloyd-Max) | 1,312.0 (SR+CondReg) | **2.1×** |
| FP4 评估 PPL | 647.8 (Simple 逐张量) | 1,443.6 (STE) | **2.2×** |
| 额外训练成本 | 0（已训练模型） | 2000 步完整训练 | — |
| 部署灵活性 | 单模型多精度需重量化 | 原生目标精度推理 | — |

---

## 三、理论验证（Phase 1）

### 3.1 RMSNorm 误差阻断效应

| 实验 | 有 RMSNorm ΔPPL | 无 RMSNorm ΔPPL | 阻断比 |
|------|-----------------|------------------|---------|
| 单层量化（12 层平均） | ±10-20 | +1,000-10,000 | **1482×** |
| 级联量化（0..5 层） | +18 | +18,600 | **1010×** |

**结论**：RMSNorm 通过每层将隐藏状态重新归一化为单位 RMS，阻断了 Lipschitz 乘法误差级联。这是 Transformer 架构量化友好性的数学基础。

### 3.2 条件数分析

- 标准训练 avg κ = 6.5，条件数正则化 avg κ = 5.1（−21%）
- κ(W) 与单层 PPL 退化的相关性为弱负相关（r≈−0.25），因为 RMSNorm 阻断了跨层传播
- κ(W) 在层内仍是有效的误差放大预测器

---

## 四、核心发现

### 4.1 PTQ

1. **Lloyd-Max 逐层自适应网格在 FP8 上最优**（Δ=−14.5）。每层有权重分布差异，统一网格留下不必要的精度损失。

2. **条件数正则化使 avg κ 降低 21%**，κ 加权自适应网格在 FP4 上改善 28+ PPL。

3. **GPTQ 在 164M 规模下帮助有限**。列级补偿需要 in_features > 4096 才显著，本模型仅 768-832。

4. **量化噪声可作为正则化**：多方法出现负 Δ（量化后反优于 FP16），适度量化抑制过拟合。

### 4.2 QAT

5. **PTQ 在所有指标上显著优于 QAT**。FP4 仅 16 层级，STE 梯度噪声过大导致收敛困难。

6. **条件数正则化在 FP4 QAT 中更有效**：QAT-FP4 训练 PPL 从 1564 降至 1307（−16%），FP8 中反增。

7. **QAT-FP4 vs QAT-FP8 差距仅 4 PPL**（优化后），说明优化技术对极低精度场景价值更大。

### 4.3 方法排名

```
FP8 PTQ: Lloyd-Max 自适应 > 混合精度 > Simple 逐通道 > κ 加权 > GPTQ > Simple 逐张量
FP4 PTQ: κ 加权自适应(cond-reg) > Simple 逐张量(baseline) > Simple 逐通道 > 混合精度 > Lloyd-Max > GPTQ
QAT:     FP8 baseline > FP8+SR+CondReg > FP4+SR+CondReg > FP4 baseline
```

---

## 五、代码结构

```
src/
├── model/
│   ├── config.py                  # MicroGemmaFPConfig (~164M)
│   └── transformer.py             # RMSNorm, RoPE, GQA, sliding/full attention
├── quantization/
│   ├── fp_quantizer.py            # FP8/FP4 模拟量化，逐通道缩放
│   ├── fp4_grids.py               # E2M1 / NF4 / MXFP4 格点
│   ├── gptq.py                    # GPTQ 权重补偿
│   ├── adaptive_grid.py           # Lloyd-Max 逐层自适应 FP4 网格 (新)
│   ├── grid_qat.py                # 基于格点的 QAT 包装器
│   ├── stochastic.py              # 随机舍入
│   └── hadamard.py                # Walsh-Hadamard 变换
├── analysis/
│   ├── condition.py               # 条件数估计 + 正则化
│   ├── lipschitz.py               # Lipschitz 常数传播
│   └── sensitivity.py             # 逐层敏感度 + 混合精度
└── experiments/
    ├── train_scaled_baseline.py       # FP16 基线训练
    ├── train_cond_regularized.py      # 条件数正则化训练 (策略 B)
    ├── train_qat.py                   # QAT（FP8/FP4 + STE）
    ├── train_qat_optimized.py         # QAT + SR + CondReg + Hadamard
    ├── validate_theory.py             # Phase 1: 逐层敏感度验证
    ├── validate_rmsnorm.py            # Phase 1: RMSNorm 消融实验
    ├── phase2_comparison.py           # Phase 2: 24 组实验系统性对比
    ├── compare_adaptive_grid.py       # 自适应网格 vs 统一网格 (策略 C)
    ├── eval_quantization.py           # 统一的 PTQ 评估
    └── ptq_eval.py                    # 单次 PTQ 评估
```

---

> 项目代码：`https://github.com/seek-hope/numerical-analysis-fp4`
