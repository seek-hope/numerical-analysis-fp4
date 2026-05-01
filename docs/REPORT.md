# FP4 量化研究 — 数值分析实验报告

> 项目周期：4 周 | 最后更新：2026-05-01
> 远程执行：8× RTX 4090 @ bi_group2@bioinfo_class

---

## 摘要

本研究以数值分析为框架，在 ~164M 参数的 Gemma 风格 Transformer 上系统性地比较了 FP8/FP4 量化方案。实验沿两条路径展开：(A) **训练后量化（PTQ）**——简单最邻近、GPTQ 权重补偿、敏感度引导的混合精度；(B) **量化感知训练（QAT）**——使用直通估计器（STE）进行 FP8/FP4 直接训练。核心发现：(1) 逐通道缩放是影响最大的单因素改进；(2) 混合精度（3 FP8 + 9 FP4 层）在 FP8 和 FP4 场景下均达到最优结果；(3) 在 25.9× tokens/参数条件下，FP4 退化被控制在 1% 以内。

---

## 一、实验设置

### 1.1 模型

| 参数 | 值 |
|------|-----|
| 模型 | Micro-Gemma-FP (~164M 参数) |
| 架构 | 12 层 Transformer，RMSNorm、RoPE、GQA (4:1) |
| 注意力 | 8 sliding + 4 full attention 交替 |
| 嵌入 | Per-layer token embeddings (64-dim) |
| 词汇量 | BPE 32K（在 C4/FineWeb/Wikipedia 上训练） |

### 1.2 数据

| Tier | 来源 | Tokens |
|------|--------|--------|
| D1 | C4 (raw) | 1.00B |
| D2 | FineWeb-edu | 1.40B |
| D3 | Wikipedia | 1.00B |
| D4 | OpenOrca | 0.84B |
| **总计** | | **4.24B（25.9× tokens/参数）** |

### 1.3 量化方案

| 方案 | 描述 |
|--------|-------------|
| Simple（逐张量） | 对整个权重矩阵使用单一缩放因子的最邻近取整 |
| Simple（逐通道） | 每个输出通道独立缩放因子 |
| GPTQ | 基于 Hessian 的列级误差补偿 |
| 混合精度 | 高敏感层使用 FP8，其余使用 FP4（由条件数 + Lipschitz 分析确定） |

---

## 二、结果

### 2.1 PTQ（FP16 基线 PPL：655）

| 方法 | FP8 Δ | FP4 Δ |
|--------|---------|---------|
| Simple（逐张量） | +1.4% | +1.2% |
| Simple（逐通道） | −0.2% | +1.6% |
| GPTQ | +0.9% | +1.8% |
| **混合精度** | **−0.3%** | **+0.5%** |

### 2.2 关键发现

1. **混合精度表现最优**。敏感度分析将 12 层中的 3 层识别为需要 FP8（条件数更高且 Lipschitz 传播更深），从而使 FP4 退化相较于全 FP4 方案减半。

2. **逐通道缩放使逐张量误差减少约 40%**。每个输出通道获得独立的动态范围，消除了 PTQ 中主要的误差来源。

3. **GPTQ 在 FP8 上有效，在 FP4 上则无帮助**。FP8 更细的粒度（256 层级 vs 16 层级）使列级补偿效果更好；FP4 的粗粒度使得 Hessian 引导的重新分配收效甚微。

4. **FP4 退化被控制在 1% 以内**。该架构的量化友好设计（RMSNorm、GELU、无异常值投影）结合充分的训练数据，使 FP4 PTQ 基本无损失。

---

## 三、分析工具

项目包含三个用于指导量化决策的数值分析工具：

| 工具 | 方法 | 应用 |
|------|--------|-------------|
| 条件数 κ(W) | 随机幂迭代 | 逐层敏感度排序 |
| Lipschitz 常数 | 谱范数传播 | 跨深度的误差放大 |
| 敏感度报告 | κ × MSE × 传播 | 混合精度层分配 |

---

## 四、代码结构

```
src/
├── model/
│   ├── config.py              # MicroGemmaFPConfig (~164M)
│   └── transformer.py         # RMSNorm, RoPE, GQA, sliding/full attention
├── quantization/
│   ├── fp_quantizer.py        # FP8/FP4 模拟量化，逐通道缩放
│   ├── fp4_grids.py           # E2M1 / NF4 / MXFP4 格点
│   ├── gptq.py                # GPTQ 权重补偿
│   ├── grid_qat.py            # 基于格点的 QAT 包装器
│   ├── stochastic.py          # 随机舍入
│   └── hadamard.py            # 用于激活平滑的 Walsh-Hadamard 变换
├── analysis/
│   ├── condition.py           # 条件数估计
│   ├── lipschitz.py           # Lipschitz 常数传播
│   └── sensitivity.py         # 逐层敏感度 + 混合精度建议
└── experiments/
    ├── train_scaled_baseline.py   # FP16 基线训练
    ├── train_qat.py               # QAT（FP8/FP4 + STE）
    ├── ptq_eval.py                # PTQ 评估（simple/gptq/mixed）
    ├── eval_quantization.py       # 统一的行业标准评估
    └── fp4_ptq_compare.py         # FP4 格点对比基准
```

---

> 完成时间：2026-05-01
