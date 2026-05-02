# 数值分析驱动的神经网络低精度量化研究

## 项目设计书

> 课程：数值分析 | 提交日期：2026-05-02

---

## 一、项目背景与研究动机

### 1.1 问题的数值分析本质

神经网络量化——将 32 位浮点权重压缩为 8 位或 4 位——本质上是一个数值逼近问题。将权重矩阵 $W$ 量化为 $\hat{W}$ 时，前向传播的输出误差满足经典矩阵扰动界：

$$\frac{\|\delta y\|}{\|y\|} \leq \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|}$$

其中 $\kappa(W) = \sigma_{\max}/\sigma_{\min}$ 是权重矩阵的条件数。这意味着：**量化引起的输出误差上界由条件数线性控制**。

这一关系直接将数值分析的核心概念——条件数、误差传播、舍入策略——与神经网络的量化性能联系起来，为项目提供了坚实的理论基础。

### 1.2 研究动机

当前工业界已广泛采用 FP8 训练（NVIDIA Transformer Engine），但 FP4（仅 16 个数值格点）仍处于前沿探索阶段。FP4 可将显存和带宽需求在 FP8 基础上再减半，但精度损失是否可控尚无定论。本研究试图回答三个递进的问题：

1. **理论问题**：条件数和误差传播理论能否定量预测量化对模型性能的影响？
2. **工程问题**：现有工业界量化方案（逐通道缩放、GPTQ 权重补偿、混合精度）在实际模型上的效果如何？
3. **方法问题**：能否利用数值分析工具设计优于现有方案的量化策略？

---

## 二、理论框架

### 2.1 核心定理

**定理 1（单层量化误差界）**。对于线性层 $y = Wx$，设量化权重 $\hat{W} = W + \delta W$。若 $\|\delta W\|$ 充分小，则输出相对误差满足：

$$\frac{\|\hat{y} - y\|}{\|y\|} \leq \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|} + O(\|\delta W\|^2)$$

其中 $\kappa(W) = \|W\| \cdot \|W^{-1}\|$（或 $\sigma_{\max}/\sigma_{\min}$ 对非方阵）。

**推论 1.1**。对于具有 $L$ 层且使用 RMSNorm 归一化的 Transformer，第 $\ell$ 层的量化误差**不跨层累积传播**，因为 RMSNorm 在每层输入处将信号重新归一化为单位 RMS，阻断了乘法误差级联。

**定理 2（RMSNorm 的误差阻断效应）**。设第 $\ell$ 层输出为 $y_\ell$，量化误差为 $\delta y_\ell$。RMSNorm 的输出为：

$$\text{RMSNorm}(y_\ell + \delta y_\ell) = \frac{y_\ell + \delta y_\ell}{\|y_\ell + \delta y_\ell\|/\sqrt{d}}$$

在 $\|\delta y_\ell\| \ll \|y_\ell\|$ 时，一阶近似下 RMSNorm 后的误差幅值为：

$$\|\delta_{\text{output}}\| \approx \frac{\|\delta y_\ell\|}{\|y_\ell\|} \cdot \sqrt{d}$$

即误差的相对幅值被限制在原层输出的相对误差水平，**不随深度指数增长**。

### 2.2 实验验证方案

本项目的独特贡献在于：不仅应用这些理论，还通过受控实验**定量验证或推翻**它们。

- **验证 1**：单层量化实验——逐层量化并测量实际 PPL 退化，与 κ(W) 预测对比
- **验证 2**：RMSNorm 消融实验——移除 RMSNorm vs 保留 RMSNorm，测量误差传播差异
- **验证 3**：舍入策略对比——确定性舍入 vs 随机舍入，验证无偏估计的 O(√n·u) 累积性质

---

## 三、研究方案

### 3.1 实验平台

| 参数 | 取值 |
|------|------|
| 模型 | ~164M 参数 Gemma 风格 Transformer |
| 架构 | 12 层，RMSNorm，RoPE，GQA (12Q/3KV) |
| 训练数据 | 4.24B tokens（C4 + FineWeb + Wikipedia + OpenOrca） |
| 数据预处理 | BPE 分词器（vocab=32000） |
| 硬件 | 8× RTX 4090 GPU |

### 3.2 三阶段研究设计

**阶段一：理论验证**

目标：通过受控实验验证或修正数值分析理论对量化误差的预测能力。

| 实验 | 方法 | 预期产出 |
|------|------|----------|
| 逐层量化敏感性 | 每次仅量化一层，测量 PPL 退化 | 验证 κ(W) 的层内预测力 |
| RMSNorm 消融 | 移除/保留 RMSNorm 下的误差传播对比 | 证明 RMSNorm 阻断跨层传播 |
| 舍入策略对比 | 确定性 vs 随机舍入，多精度 | 验证无偏估计的累积性质 |

**阶段二：工业界方案基准测试**

目标：在同一平台上系统性对比现有量化方案，测量其相对于理论最优的差距。

| 类别 | 方法 | 数值分析维度 |
|------|------|--------------|
| PTQ | 逐张量/逐通道/ GPTQ / 混合精度 | κ(W) 放大、动态范围、Hessian 补偿 |
| QAT | FP8/FP4 STE、随机舍入、Hadamard | 梯度无偏性、正交变换、条件数正则化 |

**阶段三：数值分析驱动的新策略**

目标：利用数值分析洞察设计优于现有方案的新量化策略。

**策略 A：κ 加权自适应量化网格**

核心思想：不同层的权重分布不同 → 最优量化网格也应不同。利用 Lloyd-Max 迭代为每层设计最小化 MSE 的自定义 FP4 网格，并在网格设计中融入 κ(W) 加权（高条件数层采用更保守的量化策略）。

$$\text{grid}_\ell^* = \arg\min_{g} \mathbb{E}_{w \sim W_\ell}\left[\min_{g_i \in g} (w - g_i)^2 \cdot \kappa(W_\ell)^\alpha\right]$$

**策略 B：条件数正则化训练**

在 FP16 训练中加入 $\lambda \cdot \log\kappa(W)$ 正则项，主动降低权重矩阵的条件数，使其在训练后对量化更友好。

**策略 C：混合敏感度 QAT**

仅对条件数最高的 20% 层使用 QAT（STE 训练），其余层使用 PTQ。结合 QAT 的精度优势和 PTQ 的效率。

### 3.3 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| PPL 退化 | $\text{PPL}_{\text{quant}} - \text{PPL}_{\text{fp16}}$ | 总体性能损失 |
| κ 加权误差 | $\sum_\ell \kappa(W_\ell) \cdot \text{MSE}_\ell$ | 条件数感知的量化质量 |
| 逐层 MSE | $\|\hat{W}_\ell - W_\ell\|^2 / \|W_\ell\|^2$ | 每层量化精度 |
| 理论-实际相关性 | Pearson r(预测退化, 实际退化) | 理论模型验证 |

---

## 四、预期贡献

### 4.1 理论贡献

1. **RMSNorm 误差阻断定理的定量验证**：首次通过受控实验证明 RMSNorm 如何打破 Lipschitz 乘法误差级联，为 Transformer 量化友好性提供数值分析解释。

2. **条件数-量化敏感性的统计关系**：测量 κ(W) 在不同层类型（sliding/full attention）和深度位置下对量化误差的预测力。

### 4.2 方法贡献

3. **κ 加权自适应量化网格**：将条件数融入 Lloyd-Max 量化器设计，为每层生成最优 FP4 格点的新策略。

4. **工业界方案的系统性基准**：在统一平台上对比 8 种 PTQ/QAT 方案，揭示各方法的数值分析本质差异。

### 4.3 工程贡献

5. **可复现的实验平台**：开源完整代码（模型、量化工具、分析工具、实验脚本），支持后续研究扩展。

---

## 五、时间安排

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 第 1 周 | 理论验证实验（逐层量化、RMSNorm 消融） | 3-4 天 |
| 第 2 周 | 工业界方案基准测试（PTQ + QAT 矩阵） | 4-5 天 |
| 第 3 周 | 新策略实现与评估（κ 网格、条件数正则化） | 4-5 天 |
| 第 4 周 | 综合分析、报告撰写、代码整理 | 3-4 天 |

---

## 六、参考文献

1. Frantar et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*.
2. Dettmers et al. (2024). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*.
3. Xiao et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs." *ICML 2023*.
4. Chee et al. (2024). "QuIP: 2-Bit Quantization of Large Language Models With Guarantees." *NeurIPS 2024*.
5. Golub & Van Loan (2013). *Matrix Computations*, 4th ed. Johns Hopkins University Press.
6. Higham (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.

---

> 项目代码：`https://github.com/seek-hope/numerical-analysis-fp4`
