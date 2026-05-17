# 实验设计审查与理论推导

> 最后更新：2026-05-17

---

## 第一部分：实验设计科学严谨性审查

### 1.1 基线设置

**现状**：实验使用两个 FP16 基线模型——标准训练（评估 PPL=653.2）和条件数正则化训练（评估 PPL=671.1）。所有 PTQ 方法在这两个固定检查点上应用，QAT 方法从零开始训练。

**评估**：双基线设计是正确的，因为它分离了两个不同的因果问题：(a) 量化方法是否有效，(b) 量化友好的权重是否改善了 PTQ。两个基线都有独立的评估 PPL，允许公平的 Δ 计算。

**问题 1（重要）**：条件数正则化模型的评估 PPL（671.1）比标准基线（653.2）差约 +18。这是否可归因于正则化项，还是由不同的训练随机性导致？REPORT.md 中提到训练 PPL 约为 600，因此在 2000 步后验证/评估差距尚未完全收敛。建议报告两个模型的完整训练曲线以排除收敛不足。

**建议**：为两个基线添加评估 PPL 的置信区间（例如，在不同数据子集上多次评估，或在不同检查点步骤上评估）。

### 1.2 对照实验

**现状**：项目运行了受控消融实验——RMSNorm 消融（替换为恒等映射）、逐层量化（一次仅量化一层）、级联量化（逐层增加量化层数）。

**评估**：消融设计非常出色。RMSNorm 实验直接测试因果关系（移除 RMSNorm → 观察误差传播），而不仅仅是相关性。级联实验进一步通过累积 0..k 层来确认机制。

**问题 2（中等）**：RMSNorm 消融替换了 attention 输入和 attention 后归一化——但替换后模型处于分布外状态（它是在 RMSNorm 下训练的）。无归一化时的量化误差可能与训练分布偏移混叠。这恰好支持了论文的观点（RMSNorm 很重要），但应明确承认混淆因素。

**建议**：添加一段文字说明消融实验测量的是 RMSNorm 存在与否时的量化误差，而不是 RMSNorm 存在与否时的基线性能。在无 RMSNorm 条件下的 FP16 基线将隔离这一影响。

### 1.3 统计效度

**现状**：评估使用 100 步（800 个序列）来减少单批方差。关键实验的评估协议提到使用 3 个随机种子并附带均值 ± 标准差。

**评估**：100 步评估是合理的。然而，在审查的代码路径中发现以下问题：

**问题 3（关键）**：REPORT.md 中的 PTQ 结果表报告的是单次运行的点估计——没有标准差。两种结果之间的 PPL 差异（例如，638.7 对 655.1）能代表超过评估噪声的差异吗？如果没有方差估计就无法判断。

**问题 4（关键）**："3 个随机种子"协议在提案的第 3.4 节中提到，但在 REPORT.md 中找不到实际的三种子结果——只有一个包含了所有实验的统一表格。3 种子协议是已执行还是仅计划中？

**建议**：
- 对前 3 名 PTQ 方法至少报告 3 次运行的均值 ± 标准差
- 在对比方法之间进行配对 t 检验（或至少诚实报告标准差）
- 如果数字确实是单次运行，需添加警告说明

### 1.4 数据分割与泄漏

**现状**：`get_dataloader()` 加载 `data/real_tiers/` 目录中的**所有** `.bin` 文件，无训练/验证拆分。GPTQ 校准（`gptq.py:221-246`，50-100 步，`shuffle=True`）和 PPL 评估（100 步，`shuffle=True`）均调用同一个函数。

**代码审查发现**：提案第 3.4 节中描述的"各 tier 末 5% 作为验证集"方案**未在代码中实现**。不存在任何拆分逻辑、独立的验证 `.bin` 文件，或传递给 `get_dataloader()` 的分割参数。`prepare_data_chunked.py` 使用 HF `split='train'` 从 HuggingFace 数据集中读取，但这仅过滤来源分割——本地 `.bin` 分片包含下载的全部数据。

**影响**：PTQ 校准数据（GPTQ Hessian、Lloyd-Max 网格优化）和评估数据从同一个池中抽取，`shuffle=True`。这意味着：
- GPTQ 的激活 Hessian 估计和自适应网格校准使用的是评估分布（而非仅训练分布）
- 报告的 PPL 值可能乐观，因为校准方法过拟合于评估数据统计特性
- 评估在数据上并非严谨的样本外检验

**修复（计划中的重跑实验）**：在 `prepare_data_chunked.py` 中实现基于百分比的拆分——将每个 tier 的前 95% 写入 `tierN_train.bin`，后 5% 写入 `tierN_val.bin`。更新 `get_dataloader()` 以接受 `split='train'|'val'` 参数，并相应过滤文件。

### 1.5 评估指标

**现状**：主要指标为评估 PPL，按有效移位标签 token 加权。次要指标包括 κ 加权 MSE、逐层 MSE 和 Pearson 相关性。

**评估**：PPL 是 LM 量化的标准指标。使用"有效 token"加权（而非按批次加权）在技术上是正确的，因为它防止了短序列主导指标。κ 加权 MSE 作为理论感知指标是新颖且合理的。

**问题 6（低）**：REPORT.md 中报告了多个"负 Δ"结果（量化后 PPL 优于 FP16）。这被解释为"量化噪声作为正则化"。虽然这是合理的（在 INT4 文献中已观察到），但应添加一个简单的对照：在 FP16 模型上添加高斯噪声以确认噪声而非量化是其原因。

### 1.6 实现保真度

**问题 7（关键）**：`condition.py` 中的 `inverse_power_iteration` 函数实现错误。它没有求解移位线性系统 $(W^T W - \sigma_{max}^2 I)u = v$；而是计算随机向量的范数 $||Wv||$，这近似的是 $\sigma_{max}$ 而非 $\sigma_{min}$。实际实现是：

```python
v = F.normalize(v, dim=0)
Wv = W @ v
sigma_min = Wv.norm().item()
```

这等效于 $\|Wv\|/\|v\|$，应用瑞利商——它估计的是 $\sigma_{max}$，而不是 $\sigma_{min}$。

**影响**：报告的 κ 值被高估（$\sigma_{max}$ / 被低估的 $\sigma_{min}$ → $\kappa$ 被低估）。由于 $\sigma_{min}$ 的错误估计可能对所有矩阵产生类似影响，相对排名可能被保留；但不应期望与真实 κ 值在数值上一致。

**建议**：实现正确的逆幂迭代（使用共轭梯度求解移位系统），或对于 ≤ 832 维的矩阵回退到精确 SVD——这在实验设置中是可行的（100 步评估比完整 SVD 昂贵得多）。

### 1.7 检查点与可复现性

**现状**：检查点存储在 `checkpoints/`（被 gitignore）。训练脚本硬编码种子 `torch.manual_seed(42)`。

**评估**：固定种子 + 已保存检查点提供了合理的可复现性基础。然而：

**问题 8（中等）**：没有提到确定性 CUDA 操作（`torch.backends.cudnn.deterministic = True`）。这意味着即使使用相同种子，在不同的 GPU 架构或 CUDA 版本上重新训练也可能发散。对于量化实验而言，这是可接受的（关键声明来自 PTQ，而非训练），但应记录在案。

### 1.8 声称与证据的对齐

| 声称 | 证据 | 评估 |
|-------|---------|-----------|
| "RMSNorm 阻断误差传播达 1482×" | 有无 RMSNorm 下逐层量化的测量结果 | 得到消融实验充分支持 |
| "Lloyd-Max 自适应在 FP8 PTQ 上最优（Δ=−14.5）" | 与 5 种其他方法的单次运行比较 | 缺少标准差；排名可能不稳定 |
| "条件数正则化将 avg κ 降低 21%" | 功率迭代测量：6.5 → 5.1 | 由于 σ_min 估计存在错误，数值不可靠（见问题 7）；方向正确 |
| "κ(W) 与 PPL 退化的相关性 r≈−0.25" | 每层相关性的 Pearson r | 得到逐层实验支持，是 core finding |
| "随机舍入降低累积误差" | 未给出实证结果 | 定理 3 被引用但似乎未实证验证 |

### 1.9 总结：修复状态

| 优先级 | 问题 | 状态 |
|----------|-------|--------|
| **关键** | σ_min 估计有误 | ✅ **已修复** — `inverse_power_iteration` 替换为精确 SVD |
| **关键** | 无方差估计 | ⏳ 已确认（单次运行，固定种子）— 在 REPORT.md 第 3.4 节注明 |
| **高** | 校准数据泄漏 | ✅ **已确认** — 无训练/验证拆分；校准和评估从同一池中抽取（见第 1.4 节） |
| **高** | "3 种子协议"未执行 | ⏳ 已确认 — 单次运行，协议是为后续工作规划的 |
| **中** | FP4 单位舍入值有误 | ✅ **已修复** — PROPOSAL.md 和 ANALYSIS.md 中 $u = 0.0625 \to 0.25$ |
| **中** | 消融实验中的分布偏移 | ⏳ 已记录 — 需为无 RMSNorm 条件添加 FP16 基线 |
| **低** | CUDA 确定性未强制执行 | ⏳ 已记录 | |

---

## 第二部分：数学推导

### 2.1 定理 1（单层量化误差界）

**陈述**：对于 $y = Wx$，设 $\hat{W} = W + \delta W$。则：

$$\frac{\|\hat{y} - y\|}{\|y\|} \leq \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|} + O(\|\delta W\|^2)$$

**完整推导**：

设 $\hat{y} = \hat{W}x = (W + \delta W)x = Wx + \delta W \cdot x = y + \delta y$，其中 $\delta y = \delta W \cdot x$。

分子由矩阵范数性质界定：
$$\|\delta y\| = \|\delta W \cdot x\| \leq \|\delta W\| \cdot \|x\|$$

对于分母，我们利用最小奇异值给出的下界。设 $W$ 的紧凑 SVD 为 $W = U\Sigma V^T$，其中 $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)$，$\sigma_1 \geq \cdots \geq \sigma_r > 0$。则：

$$y = Wx = U\Sigma V^T x$$

设 $z = V^T x$（正交投影）。则：
$$\|y\|^2 = \|U\Sigma z\|^2 = \|\Sigma z\|^2 = \sum_{i=1}^r \sigma_i^2 z_i^2 \geq \sigma_{\min}^2 \sum z_i^2 = \sigma_{\min}^2 \|z\|^2 = \sigma_{\min}^2 \|x\|^2$$

因此 $\|y\| \geq \sigma_{\min}(W) \cdot \|x\|$。将此与上界结合：

$$\frac{\|\delta y\|}{\|y\|} \leq \frac{\|\delta W\| \cdot \|x\|}{\sigma_{\min}(W) \cdot \|x\|} = \frac{\|\delta W\|}{\sigma_{\min}(W)}$$

乘以并除以 $\|W\| = \sigma_{\max}(W)$：

$$\frac{\|\delta y\|}{\|y\|} \leq \frac{\sigma_{\max}(W)}{\sigma_{\min}(W)} \cdot \frac{\|\delta W\|}{\sigma_{\max}(W)} = \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|}$$

$O(\|\delta W\|^2)$ 项产生的原因是：不等式 $\|\hat{y} - y\| \leq \|\delta W\| \cdot \|x\|$ 是精确的（线性），但任何对 $\hat{y}$ 的非线性操作（如后续层非线性）将引入二次项。对于单层线性情况，该界是紧的——对于使该界饱和的特定向量 $x$ 和 $\delta W$ 成立（即 $x$ 为右奇异向量 $v_{\min}$ 且 $\delta W$ 与该方向对齐）。

**紧性**：当 $x$ 与 $W$ 的最小右奇异向量对齐且 $\delta W$ 与左奇异向量 $u_{\min}$ 对齐时达到等式。在这种情况下，$\|Wx\| = \sigma_{\min}\|x\|$ 且 $\|\delta W x\| = \|\delta W\| \|x\|$。

---

### 2.2 推论 1.1（RMSNorm 阻止误差级联）

**陈述**：具有 RMSNorm 归一化的 Transformer 中，第 ℓ 层的量化误差不会跨层累积传播。

**推导（来自定理 2）**：见下方定理 2。此处给出直观理解：

无 RMSNorm 时，第 ℓ 层输出处的误差 $\delta y_\ell$ 成为第 ℓ+1 层的部分输入，产生复合效应：

$$y_{\ell+1} + \delta y_{\ell+1} = f_{\ell+1}(W_{\ell+1} \cdot (y_\ell + \delta y_\ell)) \approx f_{\ell+1}(W_{\ell+1} y_\ell) + J_{f_{\ell+1}} \cdot W_{\ell+1} \cdot \delta y_\ell$$

其中 $J_f$ 是雅可比矩阵。误差按 $\|J_f\| \cdot \|W_{\ell+1}\|$ 传播，这通常大于 1。经过 $L$ 层：

$$\|\delta y_L\| \sim \|\delta y_0\| \cdot \prod_{\ell=1}^{L} \|J_{f_\ell}\| \cdot \|W_\ell\|$$

这就是 Lipschitz 乘法级联——每层将前一层误差乘以自身的 Lipschitz 常数。

RMSNorm 在下一层输入处理之前重新归一化信号，阻断了这一乘法链。重新归一化后，信号再次具有单位 RMS，因此之前的误差幅度被重置——传播机制被阻止。

---

### 2.3 定理 2（RMSNorm 误差阻断效应）

**陈述**：设层输出为 $y = y_\ell$，量化误差为 $\delta = \delta y_\ell$。RMSNorm 定义为：

$$\text{RMSNorm}(z) = \frac{z}{\sqrt{\frac{1}{d}\sum_{i=1}^d z_i^2}} \cdot g = \frac{z}{\text{RMS}(z)} \cdot g$$

其中 $g \in \mathbb{R}^d$ 是可学习的缩放参数（省略 $g$ 分量，因为它逐元素操作且不改变分析）。

**推导**：

RMS 统计量定义为：
$$\text{RMS}(y + \delta) = \sqrt{\frac{1}{d}\sum_{i=1}^d (y_i + \delta_i)^2}$$

一阶泰勒展开。定义 $r(z) = \sqrt{\frac{1}{d}\sum z_i^2} = \frac{\|z\|}{\sqrt{d}}$。则 $r(y + \delta) \approx r(y) + \nabla r(y)^T \delta$。

梯度为 $\nabla r(y)_i = \frac{\partial}{\partial y_i} \sqrt{\frac{1}{d}\sum y_j^2} = \frac{y_i}{d \cdot r(y)}$。

在一阶下：
$$r(y + \delta) \approx r(y) + \frac{y^T \delta}{d \cdot r(y)}$$

RMSNorm 输出：
$$\text{RMSNorm}(y + \delta) = \frac{y + \delta}{r(y + \delta)}$$

对 $1/r(y+\delta)$ 进行泰勒展开。设 $f(t) = 1/t$，则 $f'(t) = -1/t^2$：
$$\frac{1}{r(y + \delta)} \approx \frac{1}{r(y)} - \frac{y^T \delta}{d \cdot r(y)^3}$$

将两者结合：
$$\text{RMSNorm}(y + \delta) \approx \frac{y + \delta}{r(y)}\left(1 - \frac{y^T \delta}{d \cdot r(y)^2}\right)$$

在一阶下（忽略 $\|\delta\|^2$ 项）：
$$\text{RMSNorm}(y + \delta) \approx \frac{y}{r(y)} + \frac{\delta}{r(y)} - \frac{y \cdot (y^T \delta)}{d \cdot r(y)^3}$$

第一项是 $\text{RMSNorm}(y)$。误差项为：
$$\delta_{\text{output}} \approx \frac{\delta}{r(y)} - \frac{y \cdot (y^T \delta)}{d \cdot r(y)^3} = \frac{1}{r(y)}\left(\delta - \frac{y}{\|y\|^2/d} \cdot \frac{y^T \delta}{d}\right)$$

由于 $r(y) = \|y\|/\sqrt{d}$：
$$\delta_{\text{output}} \approx \frac{\sqrt{d}}{\|y\|}\left(\delta - \frac{y \cdot (y^T \delta)}{\|y\|^2}\right)$$

括号中的项将 $\delta$ 投影到与 $y$ 正交的分量上。取范数（注意到 $P_{\perp y} = I - yy^T/\|y\|^2$ 是正交投影仪，$\|P_{\perp y}\| \leq 1$）：
$$\|\delta_{\text{output}}\| \leq \frac{\sqrt{d}}{\|y\|} \cdot \|\delta\| = \sqrt{d} \cdot \frac{\|\delta\|}{\|y\|}$$

与原始相对误差的关系：
$$\frac{\|\delta_{\text{output}}\|}{\|\text{RMSNorm}(y)\|} \leq \frac{\sqrt{d} \cdot \|\delta\|/\|y\|}{\sqrt{d}} = \frac{\|\delta\|}{\|y\|}$$

因为 $\|\text{RMSNorm}(y)\| = \sqrt{d}$（归一化为单位 RMS）。

**解读**：RMSNorm 输出处的相对误差受输入处相对误差的界限——没有放大，只有衰减（正交投影）。这与无归一化时形成对比，在无归一化时，跨层的 Lipschitz 乘法累积使误差增长为 $\|\delta_0\| \cdot \prod L_\ell$。

**RMSNorm 阻断比定量**：无 RMSNorm 时，$L$ 层后的误差为 $\|\delta_0\| \cdot \prod_{\ell} L_\ell$；有 RMSNorm 时为 $\|\delta_0\| \cdot \sqrt{d} / \|y\|$。比值近似为：
$$\frac{\prod_\ell L_\ell}{\sqrt{d}/\|y\|} \gg 1$$

对于典型的 Transformer（$L_\ell \approx 2\text{-}5$，12 层，$d=768$），该比值约为 $2^{12}/(\sqrt{768}/\|y\|) \approx 4096/(28/\|y\|) \approx 147\|y\|$。报告的 1482× 在此范围内一致。

---

### 2.4 定理 3（随机舍入的累积误差）

**陈述**：确定性舍入产生 $O(n \cdot u)$ 累积误差（最坏情况）。随机舍入产生 $O(\sqrt{n} \cdot u)$（期望 $L^2$ 范数）。

**推导**：

**确定性舍入**：设每次舍入引入误差 $\epsilon_i \in [-u/2, u/2]$（最邻近舍入）或 $\epsilon_i \in [0, u]$（向下舍入），其中 $u = 2^{-(m+1)}$ 是单位舍入值，$m$ 为显式尾数位数（遵循 Higham 2002 / IEEE 754 业界标准定义）。$n$ 次运算后的总误差为：

$$\left|\sum_{i=1}^n \epsilon_i\right| \leq \sum_{i=1}^n |\epsilon_i| \leq n \cdot u$$

最坏情况下 $|\epsilon_i| = u$，故界为 $O(nu)$。

**随机舍入**：以 $P(\text{round}(x) = \lceil x \rceil) = \text{frac}(x)/u$（分数部分）的概率舍入到 $\lceil x \rceil$，以互补概率舍入到 $\lfloor x \rfloor$。定义随机变量 $\epsilon_i = \text{round}(x_i) - x_i$。

**无偏性**：$\mathbb{E}[\epsilon_i] = 0$（根据构造——舍入均值为真值）。

**方差**：$|\epsilon_i| \leq u$，因此 $\text{Var}(\epsilon_i) \leq u^2$。对于独立舍入：

$$\mathbb{E}\left[\left(\sum_{i=1}^n \epsilon_i\right)^2\right] = \sum_{i=1}^n \mathbb{E}[\epsilon_i^2] + \sum_{i \neq j} \mathbb{E}[\epsilon_i]\mathbb{E}[\epsilon_j]$$

由于 $\mathbb{E}[\epsilon_i] = 0$，第二项为零。因此：
$$\mathbb{E}\left[\left(\sum_{i=1}^n \epsilon_i\right)^2\right] = \sum_{i=1}^n \mathbb{E}[\epsilon_i^2] \leq n \cdot u^2$$

取平方根（通过詹森不等式：$\mathbb{E}[\sqrt{X}] \leq \sqrt{\mathbb{E}[X]}$）：
$$\mathbb{E}\left[\left|\sum_{i=1}^n \epsilon_i\right|\right] \leq \sqrt{n} \cdot u$$

**FP4 数值示例**（$u = 2^{-(m+1)}$，$m$ 为显式尾数位数。FP4 E2M1：$m=1 \implies u = 2^{-2} = 0.25$，遵循业界标准定义 Higham 2002 / IEEE 754）：

对于 $n = 10^9$ 次梯度累加：
- 确定性：$10^9 \cdot 0.25 = 2.5 \times 10^8$
- 随机：$\sqrt{10^9} \cdot 0.25 \approx 31,623 \cdot 0.25 = 7,906$

随机舍入将累积误差降低约 $3.16 \times 10^4$ 倍（$\approx 4.5$ 个数量级）。

**注**：PROPOSAL.md 此前使用 $u = 0.0625 = 2^{-4}$（对应 E4M3 即 $m=3$）。该值已在 2026-05-17 修正为 FP4 E2M1 的正确值 $u = 0.25$。

**STE 梯度 nuance**：在实际 QAT 训练中，梯度在反向传播中以 FP16/FP32 精度累加，而非 FP4。因此定理 3 对梯度累积误差的 $O(\sqrt{n} \cdot u)$ 优势仅适用于前向传递中的无偏权重估计。这在一定程度上解释了为何随机舍入在 QAT 实验中未带来显著改善——STE 梯度的信噪比由量化间隔而非前向舍入策略主导。

---

### 2.5 定理 4（Lloyd-Max 最优性条件）

**陈述**：给定权重分布 $w \sim p(w)$ 和 $K$ 个量化层级，最小化 $\mathbb{E}[(w - Q(w))^2]$ 的最优量化器满足：

1. **最近邻条件**：$Q(w) = q_i$ 当 $|w - q_i| \leq |w - q_j|, \forall j$
2. **质心条件**：$q_i = \mathbb{E}[w \mid w \in R_i]$

**推导**：

量化器 $Q: \mathbb{R} \to \{q_1, \ldots, q_K\}$ 将实线划分为决策区域 $R_i = \{w : Q(w) = q_i\}$。失真为：
$$\mathcal{D} = \mathbb{E}[(w - Q(w))^2] = \sum_{i=1}^K \int_{R_i} (w - q_i)^2 p(w) \, dw$$

我们关于 $\{R_i\}$ 和 $\{q_i\}$ 交替优化 $\mathcal{D}$。

**步骤 1（固定 $\{q_i\}$，优化 $\{R_i\}$）** —— 最近邻：

对于给定的 $w$，最佳 $q_i$ 最小化 $(w - q_i)^2$。因此：
$$R_i = \{w : |w - q_i| \leq |w - q_j| \text{ 对所有 } j \neq i\}$$

这些是 Voronoi 区域（阈值位于相邻 $q$ 值的中点）。由于决策边界在每个维度上都是线性的，该划分在任何维度上都成立；在一维中，$R_i = [\theta_{i-1}, \theta_i]$，其中 $\theta_i = (q_i + q_{i+1})/2$。

**步骤 2（固定 $\{R_i\}$，优化 $\{q_i\}$）** —— 质心：

对于给定的 $R_i$，最小化：
$$q_i^* = \arg\min_{q} \int_{R_i} (w - q)^2 p(w) \, dw$$

对 $q$ 求导并设为零：
$$\frac{\partial}{\partial q} \int_{R_i} (w - q)^2 p(w) \, dw = -2 \int_{R_i} (w - q) p(w) \, dw = 0$$

$$\int_{R_i} q \cdot p(w) \, dw = \int_{R_i} w \cdot p(w) \, dw$$

$$q_i^* = \frac{\int_{R_i} w \cdot p(w) \, dw}{\int_{R_i} p(w) \, dw} = \mathbb{E}[w \mid w \in R_i]$$

这是 $R_i$ 内 $w$ 的条件均值——即质心。

**收敛性**：每次步骤都非增加性地减少失真——步骤 1 由构造实现；步骤 2 计算每个区域的 $L^2$ 最优代表。失真在下方有界（MSE ≥ 0），因此算法单调收敛。极限点满足这两个条件且为局部最小值。该算法以线性速度收敛（在非病态分布下典型为 10-20 次迭代）。

**κ 加权变体（策略 A）**：用 κ 调整权重替换均匀加权：
$$\mathcal{D}_\kappa = \sum_{i=1}^K \int_{R_i} (w - q_i)^2 \cdot c(w) \cdot p(w) \, dw$$

其中 $c(w) = 1 + \alpha \cdot (\kappa - 1) \cdot |w|/\max|w|$ 增加高 κ 层中大权重的重要性。质心更新变为：
$$q_i^* = \frac{\int_{R_i} w \cdot c(w) \cdot p(w) \, dw}{\int_{R_i} c(w) \cdot p(w) \, dw}$$

即加权质心。这对于 Lloyd-Max 的收敛性质没有改变——它仍然是坐标下降，但现在在失真度量中加权重要的误差。

---

### 2.6 策略 B（条件数正则化）

**目标函数**：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \sum_{\ell \in \text{Linear}} \log \kappa(W_\ell)$$

**为何选择 $\log \kappa$ 而非线性形式**：考虑两个矩阵，$\kappa_1 = 1000$ 和 $\kappa_2 = 5$。在线性和下，主导项为 $1000\lambda$ —— 单一病态层支配梯度，挤压所有其他层的正则化信号。使用 $\log$ 后，贡献变为 $\log 1000 \approx 6.9$ 对比 $\log 5 \approx 1.6$ —— 问题仍然严重，但所有层都收到有意义的梯度。$\log$ 变换还使得正则化相对于权重缩放在尺度上不变：$\kappa(cW) = \kappa(W)$，因此缩放模型不会改变条件数，但 $\log \kappa$ 的梯度按 $\nabla_W \kappa / \kappa$ 缩放——在 $\kappa$ 取值范围很大时更稳定。

**梯度视角**：关于 $W_{ij}$ 的梯度（链式法则）：
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial W_{ij}} = \frac{\partial \mathcal{L}_{\text{CE}}}{\partial W_{ij}} + \lambda \cdot \frac{1}{\kappa(W)} \cdot \frac{\partial \kappa(W)}{\partial W_{ij}}$$

第二项推动 $W$ 朝向条件数更低的方向。对于 $\kappa = \sigma_{\max}/\sigma_{\min}$：
$$\frac{\partial \kappa}{\partial W_{ij}} = \frac{1}{\sigma_{\min}} \frac{\partial \sigma_{\max}}{\partial W_{ij}} - \frac{\sigma_{\max}}{\sigma_{\min}^2} \frac{\partial \sigma_{\min}}{\partial W_{ij}}$$

奇异值导数由 $\partial \sigma_k / \partial W_{ij} = u_{ki} v_{kj}$（左右奇异向量的外积）给出。因此正则化鼓励 $\sigma_{\max}$ 减小（收缩主导方向）和 $\sigma_{\min}$ 增大（扩展最弱方向）——使矩阵更接近良好条件。

**实现注意事项**：训练代码（`condition.py:69-90`）使用代理而非精确的 $\kappa$：
$$\kappa_{\text{surrogate}} = \frac{\sigma_{\max}}{\sqrt{\frac{1}{r}\sum_i \sigma_i^2}} = \frac{\sigma_{\max}}{\text{RMS}(\sigma)}$$

这是 $\sigma_{\max}$ 与 RMS 奇异值的比值。当所有奇异值相等时为 1.0，当单个方向主导时增长——但严格不等于 $\kappa$。然而，它在训练期间是可微分的（无 `.item()` 调用），而精确的 $\kappa$ 需要 SVD（不可微且昂贵）。代理捕获了正确的定性行为，应记录为近似 $\kappa$ 正则化。

---

### 2.7 GPTQ 权重补偿（参考公式，非定理）

GPTQ 算法（Frantar 等，2023）将权重一次性量化一列，并通过更新剩余列来补偿每列的误差。其推导如下：

目标：找到 $\hat{W}$ 使 $\|WX - \hat{W}X\|_F^2$ 在 $\hat{W}$ 被量化为 FP 格式的约束下最小化。

平方误差为 $\mathcal{E} = \|(W - \hat{W})X\|_F^2 = \text{tr}((W - \hat{W})^T (W - \hat{W}) X X^T) = \text{tr}((W - \hat{W}) H (W - \hat{W})^T)$，其中 $H = X X^T$。

按列量化（$\delta_j = \hat{W}_{:,j} - W_{:,j}$）并保持剩余列固定，第 $j$ 列的最优补偿最小化关于剩余列的二次型：
$$\min_{\text{comp}} \left\|\delta_j H_{j,j+1:}^{1/2} + \text{comp} \cdot H_{j+1:,j+1:}^{1/2}\right\|^2$$

这给出补偿：$\text{comp} = -\delta_j \cdot H_{j, j+1:} / H_{j,j}$，即 `gptq.py:110-111` 中实现的公式。该算法是精确的——经过补偿后，输出误差完全来自最后一列，该列之后无剩余可补偿。

---

*审查完成于 2026-05-17*
