# 数值分析驱动的神经网络低精度量化研究

## 项目设计书

> **这是项目启动前的原始设计书（2026-05）。** 实际实验结果、修正后的理论评估和最终结论见：
> - [`REPORT.md`](REPORT.md) — 最终实验报告（含 16 配置 PTQ 对比、RMSNorm 阻断验证、定理 1 证伪）
> - [`THEOREM.md`](THEOREM.md) — 全部定理推导（整合自 ANALYSIS.md 及 thm2/3.md）
> - [`ANALYSIS.md`](ANALYSIS.md) — 实验设计审查及修复记录
>
> 关键方向修正：原始设计以 PPL 为主要指标，后续分析发现 PPL 混淆了 RMSNorm 级联效应。当前所有实验改用 `||dy||/||y||`（线性层输出空间相对误差）。定理 1（$\kappa(W)$ 预测量化误差）已被实验证伪（Pearson r = -0.23）。

---

## 一、项目背景与研究动机

### 1.1 问题的数值分析本质

神经网络量化——将 32 位浮点权重压缩为 8 位或 4 位——本质上是一个数值逼近问题。将权重矩阵 $W$ 量化为 $\hat{W}$ 时，前向传播的输出误差满足经典矩阵扰动界：

$$\frac{\|\delta y\|}{\|y\|} \leq \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|}$$

其中 $\kappa(W) = \sigma_{\max}/\sigma_{\min}$ 是权重矩阵的条件数。这意味着：**量化引起的输出误差上界由条件数线性控制**。

这一关系直接将数值分析的核心概念——条件数、误差传播、舍入策略、最优逼近——与神经网络的量化性能联系起来，为项目提供了坚实的理论基础。

### 1.2 研究动机

当前工业界已广泛采用 FP8 训练（NVIDIA Transformer Engine），但 FP4（仅 16 个数值格点）仍处于前沿探索阶段。FP4 可将显存和带宽需求在 FP8 基础上再减半，但精度损失是否可控尚无定论。本研究试图回答三个递进的问题：

1. **理论问题**：条件数和误差传播理论能否定量预测量化对模型性能的影响？
2. **工程问题**：现有工业界量化方案（逐通道缩放、GPTQ 权重补偿、混合精度）在实际模型上的效果如何？
3. **方法问题**：能否利用数值分析工具设计优于现有方案的量化策略？

### 1.3 与课程内容的关联

本项目核心使用以下数值分析工具，与本科数值分析课程教学内容直接对应：

| 课程概念 | 项目应用 |
|----------|----------|
| 条件数与矩阵扰动理论 | κ(W) 预测量化误差放大 |
| 幂迭代法（奇异值估计） | 随机幂迭代估计 σ_max/σ_min |
| 浮点数表示与舍入误差 | FP8/FP4 格点设计、IEEE 754 模拟 |
| 随机舍入与累积误差 | O(n·u) → O(√n·u) 的无偏估计 |
| Lipschitz 连续性 | 跨层误差传播分析 |
| Lloyd-Max 量化器 | 最优 FP4 网格设计 |
| 正则化与稳定性 | 条件数正则化训练 |

---

## 二、理论框架

### 2.1 核心定理

**定理 1（单层量化误差界）**。对于线性层 $y = Wx$，设量化权重 $\hat{W} = W + \delta W$。若 $\|\delta W\|$ 充分小，则输出相对误差满足：

$$\frac{\|\hat{y} - y\|}{\|y\|} \leq \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|} + O(\|\delta W\|^2)$$

其中 $\kappa(W) = \|W\| \cdot \|W^{-1}\|$（或 $\sigma_{\max}/\sigma_{\min}$ 对非方阵）。

*证明思路*：由 $\hat{y} = Wx + \delta W \cdot x$，得 $\|\hat{y} - y\| = \|\delta W \cdot x\| \leq \|\delta W\| \cdot \|x\|$。同时 $\|y\| = \|Wx\| \geq \sigma_{\min}(W) \cdot \|x\|$，故相对误差上界为 $\|\delta W\| / \sigma_{\min}(W) \cdot \|W\| / \|W\| = \kappa(W) \cdot \|\delta W\|/\|W\|$。

**推论 1.1**。对于具有 $L$ 层且使用 RMSNorm 归一化的 Transformer，第 $\ell$ 层的量化误差**不跨层累积传播**，因为 RMSNorm 在每层输入处将信号重新归一化为单位 RMS，阻断了乘法误差级联。

**定理 2（RMSNorm 的误差阻断效应）**。设第 $\ell$ 层输出为 $y_\ell$，量化误差为 $\delta y_\ell$。RMSNorm 的输出为：

$$\text{RMSNorm}(y_\ell + \delta y_\ell) = \frac{y_\ell + \delta y_\ell}{\|y_\ell + \delta y_\ell\|/\sqrt{d}}$$

在 $\|\delta y_\ell\| \ll \|y_\ell\|$ 时，一阶近似下 RMSNorm 后的误差幅值为：

$$\|\delta_{\text{output}}\| \approx \frac{\|\delta y_\ell\|}{\|y_\ell\|} \cdot \sqrt{d}$$

即误差的相对幅值被限制在原层输出的相对误差水平，**不随深度指数增长**。

**定理 3（随机舍入的累积误差界，Higham 2002）**。设对 $n$ 个值依次累加进行舍入，每次舍入单位为 $u = 2^{-(b+1)}$（$b$ 为尾数位数）：

- **确定性舍入**：累积误差为 $O(n \cdot u)$（最坏情况）
- **随机舍入**（无偏 $\mathbb{E}[\text{round}(x)] = x$）：累积误差为 $O(\sqrt{n} \cdot u)$（期望 $L^2$ 范数）

对于 FP4 E2M1（$m=1$，$u = 2^{-(m+1)} = 2^{-2} = 0.25$）的 $n = 10^9$ 次梯度累加：确定性误差量级 $\sim 2.5 \times 10^8$，随机舍入误差量级 $\sim 7.9 \times 10^3$——**降低约 4.5 个数量级**。

**定理 4（Lloyd-Max 最优量化器）**。给定权重分布 $W \sim p(w)$ 和 $K$ 个量化层级，最小化期望失真 $\mathbb{E}[(w - Q(w))^2]$ 的量化器满足：

1. **最近邻分配**：$Q(w) = q_i$ 当 $|w - q_i| \leq |w - q_j|, \forall j$
2. **质心条件**：$q_i = \mathbb{E}[w | w \in R_i]$，其中 $R_i$ 是分配给 $q_i$ 的 Voronoi 区域

Lloyd-Max 迭代算法交替执行这两步直到收敛。对于具有 16 个层级的 FP4，对每层独立应用此算法即可获得**层级最优的网格**。

### 2.2 实验验证方案

本项目的独特贡献在于：不仅应用这些理论，还通过受控实验**定量验证或修正**它们。

- **验证 1**：单层量化实验——逐层量化并测量实际 PPL 退化，与 κ(W) 预测对比
- **验证 2**：RMSNorm 消融实验——移除 RMSNorm vs 保留 RMSNorm，测量误差传播差异
- **验证 3**：舍入策略对比——确定性舍入 vs 随机舍入，验证无偏估计的 $O(\sqrt{n} \cdot u)$ 累积性质
- **验证 4**：Lloyd-Max 收敛性——监控每层迭代的失真函数下降，验证全局收敛

### 2.3 理论模型的局限性预判

为科学严谨起见，本项目预先识别理论模型可能失效的场景：

| 风险 | 应对方案 |
|------|----------|
| κ(W) 与实际敏感度相关性弱 | 测量 Pearson r，r<0.5 时退而使用 MSE 直接测量 |
| RMSNorm 阻断效应跨架构泛化失败 | 在不同归一化方案（LayerNorm/RMSNorm/无）下重复实验 |
| Lloyd-Max 在 16 个层级下陷入局部最优 | 多次随机初始化取最优，或使用 k-means++ 初始化 |
| 条件数正则化损害训练收敛 | 网格搜索 λ ∈ {0, 1e-5, 1e-4, 1e-3}，选择 PPL 退化 < 10% 的最大 λ |

---

## 三、研究方案

### 3.1 实验平台

| 参数 | 取值 |
|------|------|
| 模型 | ~164M 参数 Gemma 风格 Transformer |
| 架构 | 12 层，RMSNorm，RoPE，GQA (12Q/3KV)，8 sliding + 4 full attention |
| 隐藏维度 | 768，FFN 中间维度 3072 |
| 训练数据 | 4.24B tokens（C4 + FineWeb-edu + Wikipedia + OpenOrca） |
| 数据预处理 | BPE 分词器（vocab=32000），训练自语料库 |
| 训练步数 | 2000 steps（batch=8, seq_len=512）≈ 25.9× tokens/参数 |
| 硬件 | 8× RTX 4090 GPU |
| 随机种子 | 固定 seed=42（torch.manual_seed） |

### 3.2 三阶段研究设计

#### 阶段一：理论验证

目标：通过受控实验验证或修正数值分析理论对量化误差的预测能力。

| 实验 | 方法 | 度量 | 通过标准 |
|------|------|------|----------|
| 逐层量化敏感性 | 每次仅量化一层，测量 PPL 退化 | Pearson r(κ, ΔPPL) | r > 0.5 强相关 |
| RMSNorm 消融 | 移除/保留 RMSNorm 下的误差传播对比 | 阻断比 ΔPPL_no/ΔPPL_yes | > 100× 强阻断 |
| 舍入策略对比 | 确定性 vs 随机舍入，10 次独立运行 | 累积误差标准差与 √n 比 | 比值近似 u |
| Lloyd-Max 收敛 | 每层迭代过程中失真值监控 | 收敛步数与最终失真 | 20 步内收敛 |

#### 阶段二：工业界方案基准测试

目标：在同一平台上系统性对比现有量化方案，测量其相对于理论最优的差距。

| 类别 | 方法 | 数值分析维度 |
|------|------|--------------|
| PTQ | 逐张量最邻近 | 朴素基线，对应定理 1 的 $\|\delta W\|_\infty$ 界 |
| PTQ | 逐通道缩放 | 利用 $\kappa$ 不变性（缩放不改变条件数） |
| PTQ | GPTQ 权重补偿 | Hessian 引导的列级误差消除 |
| PTQ | 混合精度（敏感度引导） | κ × Lipschitz 排序的层级分配 |
| QAT | FP8/FP4 STE 直通估计 | 离散梯度的次梯度近似 |
| QAT | + 随机舍入 | 定理 3 的应用 |
| QAT | + Hadamard 旋转 | 正交变换降低权重峰度 |
| QAT | + 条件数正则化 | 主动降低 κ(W) |

#### 阶段三：数值分析驱动的新策略

目标：利用数值分析洞察设计优于现有方案的新量化策略。

**策略 A：κ 加权自适应量化网格**

核心思想：不同层的权重分布不同 → 最优量化网格也应不同。利用 Lloyd-Max 迭代为每层设计最小化 MSE 的自定义 FP4 网格，并在网格设计中融入 κ(W) 加权（高条件数层采用更保守的量化策略）。

$$\text{grid}_\ell^* = \arg\min_{g} \mathbb{E}_{w \sim W_\ell}\left[\min_{g_i \in g} (w - g_i)^2 \cdot w_{ij}\right]$$

其中样本权重 $w_{ij} = 1 + \alpha \cdot (\kappa(W_\ell) - 1) \cdot |w_{ij}|/\max|W_\ell|$，超参数 $\alpha \in [0, 1]$ 控制条件数加权强度。$\alpha = 0$ 退化为标准 Lloyd-Max，$\alpha = 0.5$ 为实验初值。

**策略 B：条件数正则化训练**

在 FP16 训练中加入 $\lambda \cdot \sum_\ell \log\kappa(W_\ell)$ 正则项，主动降低权重矩阵的条件数，使其在训练后对量化更友好：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \sum_{\ell \in \text{Linear}} \log\kappa(W_\ell)$$

使用 $\log$ 而非线性形式可避免单一病态层主导损失。λ 通过验证集网格搜索选取。

**策略 C：混合敏感度 QAT（待执行）**

仅对条件数最高的 20% 层使用 QAT（STE 训练），其余层使用 PTQ。结合 QAT 的精度优势和 PTQ 的效率。

### 3.3 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| PPL 退化 | $\text{PPL}_{\text{quant}} - \text{PPL}_{\text{fp16}}$ | 总体性能损失 |
| κ 加权误差 | $\sum_\ell \kappa(W_\ell) \cdot \text{MSE}_\ell$ | 条件数感知的量化质量 |
| 逐层 MSE | $\|\hat{W}_\ell - W_\ell\|^2 / \|W_\ell\|^2$ | 每层量化精度 |
| 理论-实际相关性 | Pearson r(预测退化, 实际退化) | 理论模型验证 |
| 平均条件数 | $\bar{\kappa} = \frac{1}{L}\sum_\ell \kappa(W_\ell)$ | 模型整体量化友好度 |

### 3.4 评估协议

- **评估集**：data/real_tiers 各 tier 末 5% 作为验证集（约 200M tokens，与训练集严格隔离）
- **批量评估**：100 步评估（800 个序列），消除单批方差
- **置信度**：关键对比实验使用 3 个随机种子的均值±标准差
- **公平性**：所有方法在同一 FP16 检查点上 PTQ，QAT 在同一初始化下训练

---

## 四、预期贡献

### 4.1 理论贡献

1. **RMSNorm 误差阻断定理的定量验证**：首次通过受控实验证明 RMSNorm 如何打破 Lipschitz 乘法误差级联，为 Transformer 量化友好性提供数值分析解释。预期获得阻断比 > 100×。

2. **条件数-量化敏感性的统计关系**：测量 κ(W) 在不同层类型（sliding/full attention）和深度位置下对量化误差的预测力。预期发现 κ 在层内有效（r > 0.5），跨层因 RMSNorm 失效。

3. **随机舍入累积误差的实证验证**：在 QAT 训练中测量梯度累积误差的标准差，验证 $O(\sqrt{n} \cdot u)$ 的尺度定律。

### 4.2 方法贡献

4. **κ 加权自适应量化网格**：将条件数融入 Lloyd-Max 量化器设计，为每层生成最优 FP4 格点的新策略。

5. **条件数正则化训练**：将 $\log\kappa(W)$ 作为训练损失的一部分，主动产生量化友好的权重。

6. **工业界方案的系统性基准**：在统一平台上对比 6 种 PTQ + 4 种 QAT 方案，揭示各方法的数值分析本质差异。

### 4.3 工程贡献

7. **可复现的实验平台**：开源完整代码（模型、量化工具、分析工具、实验脚本），支持后续研究扩展。

### 4.4 局限性说明

本研究范围内**未涉及**以下方向，留作后续工作：

- 激活量化（仅研究权重量化）
- 大于 1B 参数的模型（受 8× RTX 4090 算力限制）
- 多模态/视觉模型（仅文本 Transformer）
- 硬件层面 FP4 实测（仅 FP32 模拟）

---

## 五、时间安排

| 阶段 | 内容 | 预计时间 | 产出 |
|------|------|----------|------|
| 第 1 周 | 数据准备 + FP16 基线训练 + 理论验证实验 | 5-6 天 | 基线 checkpoint，Phase 1 实验结果 |
| 第 2 周 | 工业界方案基准测试（6 PTQ + 4 QAT） | 5-6 天 | 24 组实验结果，对比表 |
| 第 3 周 | 新策略实现与评估（κ 网格、条件数正则化） | 4-5 天 | 新方法在两个 checkpoint 上的对比 |
| 第 4 周 | 综合分析、报告撰写、代码整理 | 3-4 天 | REPORT.md，开源仓库 |

---

## 六、参考文献

**量化方法**

1. Frantar et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*.
2. Dettmers et al. (2024). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*.
3. Xiao et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs." *ICML 2023*.
4. Chee et al. (2024). "QuIP: 2-Bit Quantization of Large Language Models With Guarantees." *NeurIPS 2024*.
5. Micikevicius et al. (2022). "FP8 Formats for Deep Learning." *arXiv:2209.05433*.

**数值分析基础**

6. Golub & Van Loan (2013). *Matrix Computations*, 4th ed. Johns Hopkins University Press.
7. Higham (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.
8. Lloyd, S. P. (1982). "Least squares quantization in PCM." *IEEE Trans. Inf. Theory*, 28(2):129-137.
9. Max, J. (1960). "Quantizing for minimum distortion." *IRE Trans. Inf. Theory*, 6(1):7-12.

**模型架构**

10. Zhang & Sennrich (2019). "Root Mean Square Layer Normalization." *NeurIPS 2019*.
11. Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
12. Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models." *EMNLP 2023*.

---

> 项目代码：`https://github.com/seek-hope/numerical-analysis-fp4`
> 实验报告：`docs/REPORT.md`
