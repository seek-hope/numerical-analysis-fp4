# FP4 量化优化与数制设计 — 综合调研与项目方案（v5: 终版）

> 项目周期：4 周  
> 核心定位：**FP8 已是工业主流（NVIDIA Transformer Engine），前沿在于 FP4——**
> 从三个维度突破 FP4 的能力边界：PTQ 优化、QAT 优化、数制本身的设计。
>
> 本文档整合了：(1) 已完成的 5 组对比实验，(2) 三方向的文献调研，(3) 调整后的 4 周执行计划。

---

## 一、已完成实验：FP8 vs FP4 对比

### 1.1 实验设置

| 参数 | 值 |
|------|-----|
| 模型 | Micro-Gemma-FP (~15M params) |
| 架构特征 | per-layer embeddings, 4 sliding + 2 full attention |
| 数据 | 字符级 tokenizer（离线），~1.5KB 语料无限循环 |
| 训练 | 1500 steps, batch=8, seq=256, AdamW lr=3e-4 |
| 硬件 | 8× RTX 4090（单卡训练） |
| FP8 格式 | E4M3 模拟量化 |
| FP4 格式 | E2M1 模拟量化 |

### 1.2 实验结果

```
══════════════════════════════════════════════════════════════
Group                  Train PPL   Eval PPL   退化     耗时
──────────────────────────────────────────────────────────────
A0  FP16 baseline         1.00       1.00      —       91s
A1  PTQ FP8                 —        2.33    +133%    <1s
A2  PTQ FP4                 —        4.30    +330%    <1s
B1  QAT FP8               1.01       1.02      +2%    68s
B2  QAT FP4               1.01      13.86   +1286%   68s
C1  QAT FP8 + Hadamard    —          —        —      (8h+ 超时)
C2  QAT FP4 + Hadamard    —          —        —      (8h+ 超时)
══════════════════════════════════════════════════════════════
```

### 1.3 三大发现

**发现 1：FP8 QAT 几乎无损，显著优于 PTQ**
- QAT FP8 PPL 1.02（+2%） vs PTQ FP8 PPL 2.33（+133%）
- 说明：在训练时让模型感知量化约束，可以几乎消除 FP8 的精度代价

**发现 2：FP4 存在质变性的精度鸿沟**
- PTQ FP4 PPL 4.30 —— 16 个格点勉强可用但损失明显
- QAT FP4 PPL 13.86 —— 从零训练时 FP4 约束导致严重泛化失败
- 训练 PPL 1.01 但评估 PPL 13.86：经典过拟合，模型找到了"欺骗解"

**发现 3：Hadamard 旋转在 Python 中不可行**
- Python 逐元素循环导致 GPU 利用率仅 23%，训练慢 100×
- 需要用 Triton/CUDA kernel 或 `torch.compile` 实现才能在合理时间内完成

### 1.4 发现 2 的深层分析

QAT FP4 训练 PPL 1.01 → 评估 PPL 13.86 的 13.7× 泛化差距表明：
FP4 只有 16 个量化格点，优化器在如此粗糙的权重空间中只能找到"记忆但不泛化"的局部极小。
这与 Chmiel et al. (2025) 的发现一致：FP4 训练存在一个**临界梯度范数阈值**——低于该值后量化训练失效。

---

## 二、三个前沿方向调研

### 方向 1：FP4 PTQ 优化 — 压缩已有模型

**问题**：如何在不要重新训练的情况下，将 FP16 模型压缩到 FP4 且精度损失尽可能小？

**前沿方案**：

| 方法 | 核心技术 | 出处 | 可行性 |
|------|---------|------|--------|
| **DuQuant++** | Outlier-aware 细粒度旋转对齐 MXFP4 block (B=32) | 2026 | ⭐⭐⭐ 可复现 |
| **NVFP4** | E2M1 + block scale E4M3，Blackwell 原生 | NVIDIA 2025 | ⭐⭐ 需模拟 |
| **QuIP** | Hadamard 旋转 + 格点量化 | NeurIPS 2024 | ⭐⭐⭐ |

**本项目切入点**：在 FP16 基线上实现 DuQuant++ 风格的**outlier-aware 细粒度旋转**，对比标准 absmax PTQ 的精度提升。核心思路：先用异常值检测找到激活/权重中的离群通道，再对离群通道做 per-channel 缩放后再量化。

### 方向 2：FP4 QAT 优化 — 从零训练低精度模型

**问题**：如何让 FP4 QAT 训练收敛到一个可泛化的解，而不是过拟合的"欺骗解"？

**前沿方案**：

| 方法 | 核心技术 | 出处 | 可行性 |
|------|---------|------|--------|
| **FP4 All the Way** | NVFP4 格式，前向 RN + 反向随机舍入，W4A4G4 | NeurIPS 2025 | ⭐⭐⭐ |
| **Quartet** | 证明 native FP4 训练可达到全精度匹配 | NeurIPS 2025 | ⭐⭐⭐ |
| **Attn-QAT** | 前向/反向 attention 精度匹配原则 | 2026 | ⭐⭐ |
| **Metis** | 可微量化估计器替代 STE | ICML 2025 | ⭐⭐ |

**本项目切入点**：
1. 引入**随机舍入替代确定性舍入**（FP4 All the Way 已验证反向 SR 稳定训练）
2. 前向 FP4 量化 + 反向 FP16 梯度（混合精度 QAT），而非全 FP4
3. 实现**临界梯度范数自适应**：当梯度范数低于阈值时自动切换至更高精度

### 方向 3：FP8/FP4 数制设计 — 重新定义格点

**问题**：现有 FP4 E2M1 格式的 16 个格点是否最优？能否设计更适合 NN 权重的数制？

**前沿方案**：

| 方法 | 核心技术 | 出处 | 可行性 |
|------|---------|------|--------|
| **MX 微缩放** | Block of 32 共享 scale (E8M0)，FP4=MXFP4 | OCP 标准 | ⭐⭐⭐ |
| **UE5M3 scale** | 用 unsigned E5M3 作为 FP4 的 block scale | 2026 | ⭐⭐ |
| **NF4** | 16 个格点放在标准正态分位数上 | QLoRA 2023 | ⭐⭐⭐ |
| **自适应求积格点** | Clenshaw-Curtis 风格逐步加密 | 理论空白 | ⭐ |

**本项目切入点**：
1. 实现 **NF4 替代 FP4 E2M1** 作为量化格点（信息论最优 4-bit）
2. 实现 **per-channel MXFP4 风格 block scaling**（32 元素共享 scale）
3. 对比三种格点方案：FP4 E2M1 vs NF4 vs MXFP4

---

## 三、调整后的 4 周执行计划

```
Week 1: FP4 PTQ 优化
├─ Day 1-2: 复现 DuQuant++ 的 outlier-aware 细粒度旋转
│   - 实现 per-channel kurtosis 异常值检测
│   - 实现旋转块对齐 (B=32) 的细粒度 Hadamard
├─ Day 3-4: 在 FP16 基线上做 PTQ 到 NF4 / MXFP4 / FP4 E2M1
│   - 量化后 PPL 对比（三种格点方案）
│   - 逐层异常值分析：哪些层受量化影响最大
└─ Day 5-7: 报告方向 1 结论
    - DuQuant++ 风格旋转 vs 标准 PTQ 的精度提升量化

Week 2: FP4 QAT 优化
├─ Day 8-10: 实现混合精度 QAT-FP4
│   - 前向 FP4 量化（NF4 格点），反向 FP16 梯度
│   - 随机舍入替代确定性舍入
├─ Day 11-12: 临界梯度范数自适应
│   - 训练过程中监控 ||g||：低于阈值 → 切换 FP8
│   - 分析：QAT FP4 训练曲线 vs 我们的 QAT FP8 基线
└─ Day 13-14: 报告方向 2 结论
    - 混合精度 QAT + 随机舍入 + 临界范数 vs 全 FP4 QAT

Week 3: FP4 数制设计
├─ Day 15-17: 实现并对比三种 FP4 格点方案
│   - FP4 E2M1（标准）
│   - NF4（正态分位数）
│   - MXFP4（32 元素 block scale）
│   - 在同一个 FP16 模型上 PTQ 后对比 PPL
├─ Day 18-20: 自适应 block size 探索
│   - MXFP4 的 block size 敏感性分析 (8/16/32/64)
└─ Day 21: 报告方向 3 结论

Week 4: 系统集成 + 最终报告
├─ Day 22-24: 三方向综合实验矩阵
│   - 最优方案组合：FP4 PTQ(方向1) + FP4 QAT(方向2) + 格点(方向3)
├─ Day 25-27: 技术报告撰写
└─ Day 28: 代码整理 + 复现脚本
```

---

## 四、三方向的核心实现预览

### 4.1 方向 1：Outlier-aware 细粒度旋转

```python
def detect_outlier_channels(W, threshold=5.0):
    """检测权重中的异常值通道（kurtosis > threshold）"""
    kurt = kurtosis(W, dim=1)  # per-output-channel
    return kurt > threshold

def outlier_aware_scale(W, outlier_mask):
    """对异常值通道做 per-channel 缩放后量化"""
    W_scaled = W.clone()
    alpha = W.abs().mean(dim=1, keepdim=True)  # per-channel
    for ch in outlier_mask.nonzero():
        W_scaled[ch] = W[ch] / alpha[ch]
    return quantize(W_scaled), alpha
```

### 4.2 方向 2：混合精度 QAT + 随机舍入

```python
def qat_fp4_forward(weight, quantizer, stochastic=True):
    """前向 FP4 量化 + 随机舍入"""
    W_q = quantizer.quantize(weight, stochastic=stochastic)
    # 反向 STE：梯度通过量化点不变
    return W_q

def adaptive_precision_switch(grad_norm, threshold=1e-4):
    """当梯度范数低于阈值时，建议切换至 FP8"""
    if grad_norm < threshold:
        return 'fp8'  # FP4 量化失效
    return 'fp4'
```

### 4.3 方向 3：NF4 格点 vs FP4 E2M1

```python
def nf4_grid():
    """正态分位数格点（NF4）——信息论最优 4-bit"""
    from scipy.stats import norm
    probs = torch.linspace(0, 1, 17)[1:-1]  # 16 个等距分位点
    return torch.tensor(norm.ppf(probs.numpy()))

# FP4 E2M1: 16 个对数间隔的格点（硬件原生，但不匹配分布）
# NF4:     16 个正态分位数格点（信息论最优，需硬件适配）
# MXFP4:   block of 32 共享 scale 的 FP4（平衡前两者）
```

---

## 五、预期结论

基于实验数据和调研，本项目预期回答三个层次的问题：

**层次 1：FP8 已足够，FP4 是真正的前沿**
- FP8 QAT 基本解决（+2% vs FP16）
- FP4 存在质变性的精度鸿沟，是值得聚焦的方向

**层次 2：三个方向谁最有效？**
- 方向 1 (PTQ 优化)：投入最低，预期改善 FP4 PTQ PPL 从 4.30 → <3.0
- 方向 2 (QAT 优化)：混合精度 + 随机舍入预期消除泛化差距（PPL 13.86 → <5.0）
- 方向 3 (格点设计)：NF4 对 FP16→FP4 PTQ 预期比 FP4 E2M1 好 20-30%

**层次 3：组合最优方案**
- 方向 3 (NF4 格点) + 方向 2 (混合精度 QAT) = 理论上最强的 FP4 QAT 方案

---

## 六、参考文献

1. Chmiel B., et al. "FP4 All the Way: Fully Quantized Training of LLMs." NeurIPS, 2025.
2. Castro R., et al. "Quartet: Native FP4 Training Can Be Optimal for LLMs." NeurIPS, 2025.
3. Lin H., et al. "DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization." arXiv:2604.17789, 2026.
4. Dettmers T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS, 2023. (NF4)
5. "NVFP4: Native 4-bit Floating Point for Blackwell." NVIDIA, 2025.
6. "MX Microscaling Formats Specification." OCP (Open Compute Project), 2024.
7. "Is Finer Better? The Limits of Microscaling Formats." arXiv:2601.19026, 2026.
8. "Attn-QAT: 4-Bit Attention With Quantization-Aware Training." arXiv:2603.00040, 2026.

---

> **文档路径**：`~/Projects/Code/homework/Numerical_Analysis/proj/docs/numerical_analysis_training_project_proposal.md`  
> **修订时间**：2026-04-30（v5: 终版 — FP4 三方向聚焦）
