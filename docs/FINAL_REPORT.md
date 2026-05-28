# 数值分析驱动的神经网络低精度量化研究

## 摘要

本项目以一个 ~164M 参数的 Gemma 风格因果 Transformer 为载体，系统研究 FP8/FP4 权重量化中的误差行为，通过实证检验数值分析中经典理论工具。

## 一、背景与研究动机

### 1.1 量化作为数值分析问题

神经网络的量化，即将权重矩阵 $W \in \mathbb{R}^{m\times n}$ 从 FP32 压缩为 FP8 或 FP4，本质上是一个数值逼近问题。前向传播 $y = Wx$ 的输出误差服从经典矩阵扰动界

$$
\frac{\|\delta y\|}{\|y\|} \le \kappa(W)\cdot\frac{\|\delta W\|}{\|W\|},
$$

其中 $\kappa(W)=\sigma_{\max}/\sigma_{\min}$是权重矩阵的条件数。这一不等式直接把权重的量化扰动与线性映射输出的相对误差通过条件数串联起来，是本项目所有理论分析的起点。

### 1.2 FP4 的工业前沿与待解问题

FP8 训练已在 NVIDIA Transformer Engine 等系统中成熟落地，而 FP4（仅 16 个浮点格点）仍属前沿。FP4 在 FP8 基础上再将显存与带宽减半，但精度损失是否可控、量化误差是否能用经典数值分析理论预测，尚无统一结论。

本项目尝试回答三个递进问题：

- **Q1（理论）**：条件数与误差传播理论能否定量预测量化对模型的影响？
- **Q2（工程）**：现有工业量化方案在实际模型上的真实效果如何？
- **Q3（方法）**：能否依据数值分析思想，设计优于现有方案的策略？

### 1.3 与课程内容的对应

| 课程概念           | 项目应用                          |
| ------------------ | --------------------------------- |
| 条件数与矩阵扰动   | 定理 1，单层量化误差预测          |
| 浮点表示与舍入     | FP8/FP4 格点设计、IEEE 754 模拟   |
| Lipschitz 连续性   | 跨层误差传播、推论 1.1            |
| 随机舍入与累积误差 | 定理 3，$O(nu)\to O(\sqrt n u)$ |
| Lloyd-Max 量化器   | 定理 4，最优 FP4 网格             |
| 正则化与稳定性     | 条件数正则化训练                  |

---

## 二、理论框架与分析

本章首先给出本项目所依据的四个核心定理及其推导过程，随后介绍一个推论和一个策略的理论细节。

### 2.1 定理 1 单层量化误差界

**陈述。** 对于 $y = Wx$，令 $\hat{W} = W + \delta W$。则：

$$
\frac{\|\hat{y} - y\|}{\|y\|} \leq \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|} + O(\|\delta W\|^2)
$$

**推导。**

设 $\hat{y} = \hat{W}x = (W + \delta W)x = Wx + \delta W \cdot x = y + \delta y$，其中 $\delta y = \delta W \cdot x$。

分子由矩阵范数界定：

$$
\|\delta y\| = \|\delta W \cdot x\| \leq \|\delta W\| \cdot \|x\|
$$

对于分母，使用最小奇异值。设 $W = U\Sigma V^T$ 为紧奇异值分解，$\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)$，$\sigma_1 \geq \cdots \geq \sigma_r > 0$：

$$
y = Wx = U\Sigma V^T x
$$

令 $z = V^T x$（正交投影）：

$$
\|y\|^2 = \|U\Sigma z\|^2 = \|\Sigma z\|^2 = \sum_{i=1}^r \sigma_i^2 z_i^2 \geq \sigma_{\min}^2 \sum z_i^2 = \sigma_{\min}^2 \|z\|^2 = \sigma_{\min}^2 \|x\|^2
$$

因此 $\|y\| \geq \sigma_{\min}(W) \cdot \|x\|$。结合上界：

$$
\frac{\|\delta y\|}{\|y\|} \leq \frac{\|\delta W\| \cdot \|x\|}{\sigma_{\min}(W) \cdot \|x\|} = \frac{\|\delta W\|}{\sigma_{\min}(W)}
$$

乘以并除以 $\|W\| = \sigma_{\max}(W)$：

$$
\frac{\|\delta y\|}{\|y\|} \leq \frac{\sigma_{\max}(W)}{\sigma_{\min}(W)} \cdot \frac{\|\delta W\|}{\sigma_{\max}(W)} = \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|}
$$

$O(\|\delta W\|^2)$ 项的出现是因为不等式 $\|\hat{y} - y\| \leq \|\delta W\| \cdot \|x\|$ 是精确的（线性），但对 $\hat{y}$ 的任何非线性操作（例如后续层的非线性）会引入二阶项。对于单层线性情况，界是紧的——当 $x$ 与 $W$ 的最小右奇异向量 $v_{\min}$ 对齐且 $\delta W$ 与左奇异向量 $u_{\min}$ 对齐时，等号成立。

**紧性。** 当 $x = v_{\min}$ 且 $\delta W \propto u_{\min} v_{\min}^T$ 时，等号成立。此时 $\|Wx\| = \sigma_{\min}\|x\|$ 且 $\|\delta W x\| = \|\delta W\| \|x\|$。

### 2.2 定理 2 RMSNorm 误差阻断

**陈述。** RMSNorm 衰减输入扰动。对于输入 $y$ 和扰动 $\delta$，输出扰动满足：

$$
\frac{\|\delta_{\text{out}}\|}{\|\text{RMSNorm}(y)\|} \leq \frac{\|\delta\|}{\|y\|}
$$

无放大，仅通过正交投影进行衰减。

#### 推导 A：雅可比矩阵

RMSNorm 定义为：

$$
\text{RMSNorm}(x) = \sqrt{d} \cdot \frac{x}{\|x\|} \odot \gamma
$$

其中 $\gamma \in \mathbb{R}^d$ 是可学习的缩放参数（由于不影响分析，下文省略）。

定义 $f(x) = \sqrt{d} \cdot \frac{x}{\|x\|}$。对于小扰动 $\delta x$（$\|\delta x\| \ll \|x\|$），一阶泰勒展开给出：

$$
f(x + \delta x) \approx f(x) + J(x) \delta x
$$

雅可比矩阵 $J(x)$ 为：

$$
J(x) = \frac{\partial}{\partial x} \left( \frac{\sqrt{d}}{\|x\|} x \right) = \sqrt{d} \left( \frac{1}{\|x\|} I - \frac{x x^T}{\|x\|^3} \right) = \frac{\sqrt{d}}{\|x\|} \left( I - \frac{x x^T}{\|x\|^2} \right)
$$

注意 $P_{\perp x} = I - \frac{x x^T}{\|x\|^2}$ 是投影矩阵——它保留垂直于 $x$ 的分量并丢弃沿 $x$ 的分量。

$$
\delta_{\text{out}} \approx J(y) \delta = \frac{\sqrt{d}}{\|y\|} \left( I - \frac{y y^T}{\|y\|^2} \right) \delta = \frac{\sqrt{d}}{\|y\|} \cdot P_{\perp y} \delta
$$

取范数（因为 $\|P_{\perp y}\| \leq 1$）：

$$
\|\delta_{\text{out}}\| \leq \frac{\sqrt{d}}{\|y\|} \cdot \|\delta\|
$$

由于 $\|\text{RMSNorm}(y)\| = \sqrt{d}$（标准化为单位 RMS）：

$$
\frac{\|\delta_{\text{out}}\|}{\|\text{RMSNorm}(y)\|} \leq \frac{\sqrt{d} \cdot \|\delta\|/\|y\|}{\sqrt{d}} = \frac{\|\delta\|}{\|y\|}
$$

输出相对误差由输入相对误差界定，因此 RMSNorm 无法放大误差。

#### 推导 B：RMS 泰勒展开

定义 $r(z) = \sqrt{\frac{1}{d}\sum z_i^2} = \frac{\|z\|}{\sqrt{d}}$。梯度为 $\nabla r(y)_i = \frac{y_i}{d \cdot r(y)}$。

一阶展开：

$$
r(y + \delta) \approx r(y) + \frac{y^T \delta}{d \cdot r(y)}
$$

对于倒数 $1/r$：

$$
\frac{1}{r(y + \delta)} \approx \frac{1}{r(y)} - \frac{y^T \delta}{d \cdot r(y)^3}
$$

RMSNorm 输出：

$$
\text{RMSNorm}(y + \delta) = \frac{y + \delta}{r(y + \delta)} \approx \frac{y + \delta}{r(y)}\left(1 - \frac{y^T \delta}{d \cdot r(y)^2}\right)
$$

一阶（舍弃 $\|\delta\|^2$）：

$$
\text{RMSNorm}(y + \delta) \approx \frac{y}{r(y)} + \frac{\delta}{r(y)} - \frac{y \cdot (y^T \delta)}{d \cdot r(y)^3}
$$

第一项是 $\text{RMSNorm}(y)$。误差项为：

$$
\delta_{\text{output}} \approx \frac{1}{r(y)}\left(\delta - \frac{y \cdot (y^T \delta)}{\|y\|^2}\right) = \frac{\sqrt{d}}{\|y\|} \cdot P_{\perp y} \delta
$$

得到一致的结果。

#### RMSNorm 阻止比率

无 RMSNorm：$L$ 层后的误差 = $\|\delta_0\| \cdot \prod_{\ell} L_\ell$。
有 RMSNorm：$L$ 层后的误差 = $\|\delta_0\| \cdot \sqrt{d} / \|y\|$。

比率：

$$
\frac{\prod_\ell L_\ell}{\sqrt{d}/\|y\|} \gg 1
$$

理论推导得：对于典型 Transformer（$L_\ell \approx 2\text{--}5$，12 层，$d = 768$）：

$$
\frac{2^{12}}{\sqrt{768}/\|y\|} \approx \frac{4096}{28/\|y\|} \approx 147\|y\|
$$

### 2.3 推论 1 误差不跨层级联

**陈述。** 在使用 RMSNorm 的 Transformer 中，第 $\ell$ 层的量化误差不会跨层累积传播。

**推导。** 没有 RMSNorm 时，第 $\ell$ 层的误差 $\delta y_\ell$ 会传播到第 $\ell+1$ 层，产生复合效应：

$$
y_{\ell+1} + \delta y_{\ell+1} = f_{\ell+1}(W_{\ell+1} \cdot (y_\ell + \delta y_\ell)) \approx f_{\ell+1}(W_{\ell+1} y_\ell) + J_{f_{\ell+1}} \cdot W_{\ell+1} \cdot \delta y_\ell
$$

其中 $J_f$ 是雅可比矩阵。误差按 $\|J_f\| \cdot \|W_{\ell+1}\|$ 传播，通常 > 1。在 $L$ 层后：

$$
\|\delta y_L\| \sim \|\delta y_0\| \cdot \prod_{\ell=1}^{L} \|J_{f_\ell}\| \cdot \|W_\ell\|
$$

这是 Lipschitz 乘法级联，每一层都将前一层的误差乘以其自身的 Lipschitz 常数。RMSNorm 在下一层处理前重新标准化信号，打破这个乘法链。重新标准化后，信号再次具有单位 RMS，因此先前的误差幅度被重置，传播机制被阻止。

### 2.4 定理 3 随机舍入累积误差

**陈述。** 确定性舍入产生 $O(n \cdot u)$ 累积误差（最坏情况）。随机舍入产生 $O(\sqrt{n} \cdot u)$（期望 $L^2$ 范数）。

#### 推导 A：一般框架

**确定性舍入。** 每个舍入操作引入误差 $\epsilon_i \in [-u/2, u/2]$（四舍五入）或 $\epsilon_i \in [0, u]$（向下舍入），其中 $u = 2^{-(m+1)}$ 是单位舍入误差（IEEE 754 / Higham 2002），$m$ 是显式尾数位。$n$ 次操作后的总误差：

$$
\left|\sum_{i=1}^n \epsilon_i\right| \leq \sum_{i=1}^n |\epsilon_i| \leq n \cdot u = O(nu)
$$

**随机舍入。** 以概率 $\text{frac}(x)/u$ 舍入到 $\lceil x \rceil$，否则舍入到 $\lfloor x \rfloor$。定义 $\epsilon_i = \text{round}(x_i) - x_i$。

- **无偏性：** $\mathbb{E}[\epsilon_i] = 0$（构造得出——舍入均值等于真值）。
- **方差：** $|\epsilon_i| \leq u$，因此 $\text{Var}(\epsilon_i) \leq u^2$。对于独立舍入：

$$
\mathbb{E}\left[\left(\sum_{i=1}^n \epsilon_i\right)^2\right] = \sum_{i=1}^n \mathbb{E}[\epsilon_i^2] + \sum_{i \neq j} \mathbb{E}[\epsilon_i]\mathbb{E}[\epsilon_j]
$$

由于 $\mathbb{E}[\epsilon_i] = 0$，交叉项消失：

$$
\mathbb{E}\left[\left(\sum_{i=1}^n \epsilon_i\right)^2\right] = \sum_{i=1}^n \mathbb{E}[\epsilon_i^2] \leq n \cdot u^2
$$

取平方根（Jensen 不等式：$\mathbb{E}[\sqrt{X}] \leq \sqrt{\mathbb{E}[X]}$）：

$$
\mathbb{E}\left[\left|\sum_{i=1}^n \epsilon_i\right|\right] \leq \sqrt{n} \cdot u = O(\sqrt{n}u)
$$

#### 推导 B：部分和框架

考虑累积和 $S_n = \sum_{i=1}^n x_i$。设 $s_k$ 为第 $k$ 步的计算结果：

$$
s_k = r(s_{k-1} + x_k) = (s_{k-1} + x_k)(1 + \delta_k)
$$

每次加法的绝对误差：$\epsilon_k = (s_{k-1} + x_k)\delta_k$，其中 $|\delta_k| \leq u$。

**确定性（最坏情况）：** 所有 $\delta_k = u$ 同向 → $\|E_n\|_{\infty} = O(nu)$。

**随机：** $\delta_k$ 是随机变量，$\mathbb{E}[\delta_k] = 0$。假设步间舍入独立：

$$
\mathbb{E}[E_n^2] = \text{Var}\left( \sum_{k=1}^n \epsilon_k \right) = \sum_{k=1}^n \text{Var}(\epsilon_k)
$$

根据 Popoviciu 不等式，对于有界零均值 $\delta_k$，存在 $c$ 使得 $\text{Var}(\delta_k) \leq c u^2$。

$$
\text{Var}(\epsilon_k) = (s_{k-1} + x_k)^2 \text{Var}(\delta_k) \leq (s_{k-1} + x_k)^2 c u^2
$$

上界：任何部分和幅度 $\leq X_{\text{sum}} = \sum |x_i|$。因此：

$$
\|E_n\| = \sqrt{\mathbb{E}[E_n^2]} \leq \sqrt{c}\,\sqrt{n}\,u\,X_{\text{sum}} = O(\sqrt{n}u)
$$

#### FP4 数值示例

FP4 E2M1：$m = 1 \implies u = 2^{-(1+1)} = 0.25$。

对于 $n = 10^9$ 梯度累积：

| 方法   | 误差界               | 值                  |
| ------ | -------------------- | ------------------- |
| 确定性 | $n \cdot u$        | $2.5 \times 10^8$ |
| 随机   | $\sqrt{n} \cdot u$ | $\approx 7,906$   |

随机舍入将累积误差减少 ~$3.16 \times 10^4$×（~4.5 数量级）。

#### STE 梯度细节

在 QAT 训练中，梯度在反向传播中以 FP16/FP32 精度累积，而非 FP4。因此，定理3的 $O(\sqrt{n} \cdot u)$ 优势仅适用于前向传播的无偏权重估计。这部分解释了为什么随机舍入在 QAT 实验中未显示显著改进：STE 梯度信噪比由量化间隔主导，而非前向舍入策略。

### 2.5 定理 4 Lloyd-Max 量化器最优性

**陈述。** 给定权重分布 $w \sim p(w)$ 和 $K$ 个量化级，最小化 $\mathbb{E}[(w - Q(w))^2]$ 的量化器满足：

1. **最近邻条件：** 当 $|w - q_i| \leq |w - q_j|$ 对所有 $j$ 时，$Q(w) = q_i$
2. **质心条件：** $q_i = \mathbb{E}[w \mid w \in R_i]$

**推导。**

量化器 $Q: \mathbb{R} \to \{q_1, \ldots, q_K\}$ 将实线分割为决策区域 $R_i = \{w : Q(w) = q_i\}$。失真：

$$
\mathcal{D} = \mathbb{E}[(w - Q(w))^2] = \sum_{i=1}^K \int_{R_i} (w - q_i)^2 p(w) \, dw
$$

在 $\{R_i\}$ 和 $\{q_i\}$ 上交替优化 $\mathcal{D}$。

**步骤 1（固定 $\{q_i\}$，优化 $\{R_i\}$）— 最近邻：**

对于给定的 $w$，最佳 $q_i$ 最小化 $(w - q_i)^2$。因此：

$$
R_i = \{w : |w - q_i| \leq |w - q_j| \text{ 对所有 } j \neq i\}
$$

这些是 Voronoi 区域（阈值在相邻 $q$ 值的中点）。一维中：$R_i = [\theta_{i-1}, \theta_i]$，其中 $\theta_i = (q_i + q_{i+1})/2$。

**步骤 2（固定 $\{R_i\}$，优化 $\{q_i\}$）— 质心：**

对于给定的 $R_i$，最小化：

$$
q_i^* = \arg\min_{q} \int_{R_i} (w - q)^2 p(w) \, dw
$$

对 $q$ 求导并设为零：

$$
\frac{\partial}{\partial q} \int_{R_i} (w - q)^2 p(w) \, dw = -2 \int_{R_i} (w - q) p(w) \, dw = 0
$$

$$
\int_{R_i} q \cdot p(w) \, dw = \int_{R_i} w \cdot p(w) \, dw
$$

$$
q_i^* = \frac{\int_{R_i} w \cdot p(w) \, dw}{\int_{R_i} p(w) \, dw} = \mathbb{E}[w \mid w \in R_i]
$$

这是 $R_i$ 内 $w$ 的条件均值——质心。

**收敛性。** 每一步非增地减少失真——步骤 1 按构造；步骤 2 计算每个区域的 $L^2$ 最优代表。失真下界为 0，所以算法单调收敛。极限点满足两个条件，是局部最小值。收敛是线性的（典型：10–20 次迭代，对于行为良好的分布）。

**κ 加权变体（策略 A）。** 用 κ 调整权重替换均匀权重：

$$
\mathcal{D}_\kappa = \sum_{i=1}^K \int_{R_i} (w - q_i)^2 \cdot c(w) \cdot p(w) \, dw
$$

其中 $c(w) = 1 + \alpha \cdot (\kappa - 1) \cdot |w|/\max|w|$ 在高 κ 层中上权重化大权值。质心更新变为：

$$
q_i^* = \frac{\int_{R_i} w \cdot c(w) \cdot p(w) \, dw}{\int_{R_i} c(w) \cdot p(w) \, dw}
$$

这是加权质心。收敛性质保持不变——它仍是坐标下降，现在在失真度量中权重化重要误差。

### 2.6 策略 B 条件数正则化

**目标：**

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \sum_{\ell \in \text{Linear}} \log \kappa(W_\ell)
$$

**为什么用 $\log \kappa$ 而非线性？** 考虑两个矩阵：$\kappa_1 = 1000$，$\kappa_2 = 5$。在线性和下，$1000\lambda$ 主导——单个病态矩阵压倒所有其他层的正则化信号。用 $\log$：$\log 1000 \approx 6.9$ vs $\log 5 \approx 1.6$——仍重度加权，但所有层接收有意义的梯度。$\log$ 变换还使正则化对权重缩放不变：$\kappa(cW) = \kappa(W)$，所以缩放模型不改变条件数，但 $\log \kappa$ 的梯度缩放为 $\nabla_W \kappa / \kappa$——在 κ 值的广泛范围内更稳定。

**梯度透视。** 根据链式法则：

$$
\frac{\partial \mathcal{L}_{\text{total}}}{\partial W_{ij}} = \frac{\partial \mathcal{L}_{\text{CE}}}{\partial W_{ij}} + \lambda \cdot \frac{1}{\kappa(W)} \cdot \frac{\partial \kappa(W)}{\partial W_{ij}}
$$

第二项将 $W$ 推向更低条件数。对于 $\kappa = \sigma_{\max}/\sigma_{\min}$：

$$
\frac{\partial \kappa}{\partial W_{ij}} = \frac{1}{\sigma_{\min}} \frac{\partial \sigma_{\max}}{\partial W_{ij}} - \frac{\sigma_{\max}}{\sigma_{\min}^2} \frac{\partial \sigma_{\min}}{\partial W_{ij}}
$$

奇异值导数：$\partial \sigma_k / \partial W_{ij} = u_{ki} v_{kj}$（左/右奇异向量的外积）。正则化因此鼓励 $\sigma_{\max}$ 收缩（压缩主导方向）和 $\sigma_{\min}$ 增长（扩展最弱方向）——将矩阵推向条件良好。

**实现细节。** 训练代码使用代理而非精确 κ：

$$
\kappa_{\text{surrogate}} = \frac{\sigma_{\max}}{\sqrt{\frac{1}{r}\sum_i \sigma_i^2}} = \frac{\sigma_{\max}}{\text{RMS}(\sigma)}
$$

$\sigma_{\max}$ 与 RMS 奇异值的比。所有奇异值相等时等于 1.0，单一方向主导时增长——但不严格是 κ。然而，它在训练期间可微分（无 `.item()` 调用），而精确 κ 需要 SVD（不可微且昂贵）。代理捕获正确的定性行为，应记录为近似 κ 正则化。

## 2.7 GPTQ 权重补偿

在本研究中，我们也讨论了 GPTQ（Frantar et al., 2022）方法。这一方法逐列量化权重，并通过更新剩余列补偿每列的误差。

**目标：** 在 $\hat{W}$ 为 FP 格式的约束下，找到最小化 $\|WX - \hat{W}X\|_F^2$ 的 $\hat{W}$。

平方误差：

$$
\mathcal{E} = \|(W - \hat{W})X\|_F^2 = \text{tr}((W - \hat{W})^T (W - \hat{W}) X X^T) = \text{tr}((W - \hat{W}) H (W - \hat{W})^T)
$$

其中 $H = X X^T$ 是激活 Gram 矩阵。

逐列量化（$\delta_j = \hat{W}_{:,j} - W_{:,j}$），保持剩余列固定。列 $j$ 的最优补偿最小化关于剩余列的二次形：

$$
\min_{\text{comp}} \left\|\delta_j H_{j,j+1:}^{1/2} + \text{comp} \cdot H_{j+1:,j+1:}^{1/2}\right\|^2
$$

解：

$$
\text{comp} = -\delta_j \cdot \frac{H_{j, j+1:}}{H_{j,j}}
$$

补偿后，输出误差完全来自最后一列（无剩余列可补偿）。

## 三、实验方法与设计

### 3.1 模型与训练数据

我们选取了自实现的模型 Micro-Gemma-FP（参考 `src/model/transformer.py`），核心参数如下：

| 参数     | 取值                                                       |
| -------- | --------------------------------------------------------- |
| 参数量   | ~164M                                                      |
| 层数     | 12（其中 8 层 sliding window，4 层 full attention）         |
| 隐藏维度 | 768，FFN 中间维度 3072                                      |
| 注意力头 | 12 query 头，3 KV 头（GQA 4:1）                             |
| 归一化   | 输入端 RMSNorm + 注意力后 RMSNorm + Q/K RMSNorm             |
| 位置编码 | RoPE                                                       |
| 词表     | BPE 32K，自训练                                             |
| 训练数据 | 4.24B tokens，4 个 tier（C4、FineWeb-edu、Wikipedia、OpenOrca）|
| 训练步数 | 2000 步（batch=8, seq_len=512）                             |
| 随机种子 | seed=42（torch.manual_seed）                                |

每个 tier 的后 5% 写入独立的 `tierN_val.bin` 文件作为评估集，校准（GPTQ Hessian、Lloyd-Max 网格拟合）只读 `tierN_train.bin`，评估只读 `tierN_val.bin`，由 `get_dataloader(split='train'|'val')` 在 dataloader 层强制隔离。

两个 FP16 检查点为：

- **fp16_baseline**：标准交叉熵训练；
- **cond_regularized**：在 CE 损失上叠加 $\lambda\sum_\ell\log\kappa(W_\ell)$ 正则项（实现于 `src/analysis/condition.py`），训练目标在于产出量化更友好的权重。

### 3.2 量化方案

模拟量化全部以 FP32 实现（无硬件 FP4 执行路径）。涉及的 PTQ 方法：

| 方法                        | Code                                     | 简述                              |
| --------------------------- | ---------------------------------------- | --------------------------------- |
| RTN (round-to-nearest) | `src/quantization/fp_quantizer.py` | 逐通道缩放后取最近格点 |
| GPTQ  | `src/quantization/gptq.py` | 列序贪心量化 + Hessian 加权列补偿 |
| Lloyd-Max | `src/quantization/adaptive_grid.py` | 每层自适应 16 层级（FP4） |
| MXFP4 | `src/quantization/fp4_grids.py` | 块缩放（block_size=32）+ E2M1 |
| Hadamard 旋转 | `src/quantization/hadamard.py` | Walsh-Hadamard 旋转后量化  |
| Outlier 旋转 | `src/quantization/outlier_rotation.py` | 离群感知缩放 + Hadamard |

格式覆盖 FP8 E4M3 与 FP4 E2M1 两种。两个检查点 × 两种格式 × 四种主方法，共计 $16$ 配置横向比较；其余方法（Hadamard、outlier、MXFP4）作为对照单独列出。

### 3.3 实验指标

我们选取了如下核心指标：

1. 单矩阵相对误差均值

    $$
    \overline{\frac{\|\delta y\|}{\|y\|}} = \frac{1}{|\mathcal M|}\sum_{W\in\mathcal M}\frac{\|(\hat W - W)X\|_F}{\|WX\|_F},
    $$

    对 72 个权重矩阵均权平均。这是定理 1 左边量的直接对应：单层 Linear 输出处的相对误差。

2. 激活加权总误差

    $$
    \frac{\|\Delta WX\|}{\|WX\|} = \frac{\sqrt{\sum_W\|(\hat W - W)X\|_F^2}}{\sqrt{\sum_W\|WX\|_F^2}}.
    $$

    按每个矩阵自身的 $\|WX\|$ 加权，与 GPTQ 隐含的 Hessian 加权目标 $\min\|(W_q-W)X\|_F^2$ 对齐。这一指标让 GPTQ 这类按 Hessian 加权优化的方法不被单一等权指标所系统性影响。

3. 同时，我们也通过一些指标对某些特定场景辅助验证：如紧致比 $\tau = \dfrac{\|\delta y\|/\|y\|}{\kappa(W)\cdot\|\delta W\|/\|W\|}$（衡量定理 1 上界的紧致程度）、RMSNorm 层级阻断比 $\rho_\ell$、Pearson 相关 $r(\kappa,\|\delta y\|/\|y\|)$。

### 3.4 测量方法

本节概述实现中的测量流程，包括激活捕获、误差计算、条件数估计与统计检验的具体做法。

1. 激活捕获机制。使用 `ErrorPropagationTracker` 类（`src/analysis/error_propagation.py`）在每个 `nn.Linear` 模块上注册 `forward_pre_hook`，捕获前向传播中的输入张量 $x$。Hook在每个 Linear 层执行前触发，保证所有量化配置共用同一组 FP16 激活作为参照输入，避免抽样级联混淆。

2. 误差计算。对每个权重矩阵 $W$，离线计算输出空间相对误差：

    $$\frac{\|\delta y\|}{\|y\|} = \frac{\|(W_q - W)x\|_F}{\|Wx\|_F},$$

    其中 $x$ 是上述捕获的激活，$W$ 是 FP16 权重，$W_q$ 是量化后权重。实现中，对RTN等无状态方法，直接调用`quantizer.quantize()`来计算；对于GPTQ/Lloyd-Max这类原地修改的方法，先保存 FP16 权重副本，再调用 `quantizer.quantize()`，最后计算误差时用 FP16 权重副本与量化后权重做差。

3. 条件数计算。用 `torch.linalg.svdvals(W)` 精确 SVD 计算 $\kappa(W) = \sigma_{\max}/\sigma_{\min}$（`src/analysis/condition.py:20-40`）。本项目矩阵最大维度为 3072×768，精确 SVD 的开销远小于 100 步评估批次，无需幂迭代近似。

4. 测量粒度。 $\|\delta y\|/\|y\|$ 在每个 Linear 层的输出处直接采集（即 $y = Wx$ 之后、RMSNorm/激活/残差之前），保证粒度与单层线性映射一致，不使用 PPL 等下游混淆指标。

此外，在保证严谨性上，采用多种子重复实验、多重检验校正、bootstrap 置信区间等方法，确保实验结果的可靠性。我们也保证了校准/评估数据分离，避免 in-sample 偏差。

## 参考文献

1. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323.
2. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36, 10088-10115.
3. Shao, X. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs. In ICML.
4. Chee, J., Cai, Y., Kuleshov, V., & De Sa, C. M. (2023). Quip: 2-bit quantization of large language models with guarantees. Advances in neural information processing systems, 36, 4396-4429.
5. Micikevicius, P., Stosic, D., Burgess, N., Cornea, M., Dubey, P., Grisenthwaite, R., ... & Wu, H. (2022). Fp8 formats for deep learning. arXiv preprint arXiv:2209.05433.
6. Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU press.
7. Higham, N. J. (2002). Accuracy and stability of numerical algorithms. Society for industrial and applied mathematics.
8. Lloyd, S. (1982). Least squares quantization in PCM. IEEE transactions on information theory, 28(2), 129-137.
9. Max, J. (1960). Quantizing for minimum distortion. IRE Transactions on Information Theory, 6(1), 7-12.
10. Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. Advances in neural information processing systems, 32.
11. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2024). Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568, 127063.
12. Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023, December). Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (pp. 4895-4901).
