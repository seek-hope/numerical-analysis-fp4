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

上面的计算中，$\kappa(W)$ 基于规范界，假设扰动 $\delta W$ 可以取任意方向，这对应了最坏情况输入。下面这种建模则与实际量化噪声更加接近：由于 FP 量化的本质是每个权重元素独立舍入到最近格点，产生的是**逐元素有界的结构化扰动**；针对这一结构，Skeel、Oettli–Prager 及 Higham 框架给出了更匹配的**逐分量条件数（component-wise condition number）**。

**陈述。** 对于 $y = Wx$，设 FP 量化 $\hat{W}$ 满足逐元素相对误差不超过单位舍入 $u$：

$$
|\hat{W}_{ij} - W_{ij}| \leq u \cdot |W_{ij}|, \quad \forall i, j
$$

则前向误差满足：

$$
\frac{\|\delta y\|}{\|y\|} \leq \text{cond}_{\text{cw}}(W, x) \cdot u
$$

其中**逐分量条件数**定义为：

$$
\text{cond}_{\text{cw}}(W, x) = \frac{\| |W| \cdot |x| \|}{\|Wx\|}
$$

$|W|$、$|x|$ 表示逐元素取绝对值。

**推导。** FP 量化的逐分量后向误差为：

$$
\omega(\Delta W) = \min\{\epsilon : |\Delta W_{ij}| \leq \epsilon \cdot |W_{ij}|,\ \forall i,j\} \leq u
$$

由 Oettli–Prager 定理，前向误差界为：

$$
\|\delta y\| = \|\Delta W \cdot x\| = \left\|\sum_j \Delta W_{*,j} \cdot x_j\right\| \leq \sum_j \|\Delta W_{*,j}\| \cdot |x_j| \leq \sum_j u \cdot \||W_{*,j}|\| \cdot |x_j| = u \cdot \| |W| \cdot |x| \|
$$

两边除以 $\|y\| = \|Wx\|$ 即得该界。

$\text{cond}_{\text{cw}}(W,x)$ 通过 $|W| \cdot |x|$ 直接编码每个权重元素对输出的实际加权贡献，因此与 FP 量化的逐元素误差结构天然吻合。

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

### 2.6 策略之 κ 加权 Lloyd-Max

**动机。** 标准 Lloyd-Max（定理 4）最小化 $\mathbb{E}[(w - Q(w))^2]$，对所有权重一视同仁。然而，在高条件数矩阵（$\kappa \gg 1$）中，大幅值权重的误差在不良奇异方向上被放大更多。策略 A 通过向失真泛函引入 $\kappa$ 相关的重要性权重 $c(w; \kappa)$，为高 $\kappa$ 层的大权重分配更多量化精度。

**定义 1（κ 加权失真）。** 对于条件数为 $\kappa = \kappa(W)$、权重函数 $c: \mathbb{R} \times [1, \infty) \to \mathbb{R}^+$ 的权重矩阵 $W$，κ 加权失真定义为：

$$
\mathcal{D}_\kappa = \mathbb{E}\left[c(w; \kappa) \cdot (w - Q(w))^2\right]
$$

**定义 2（κ 权重函数，策略 A）。** 权重函数为：

$$
c(w; \kappa) = 1 + \alpha \cdot \max(0, \kappa - 1) \cdot \frac{|w|}{\max|W|}
$$

其中 $\alpha \in [0, 1]$ 为可调超参数（$\alpha = 0$ 退化为标准 Lloyd-Max；实验中取 $\alpha = 0.5$）。该函数满足：

- $c(|w|; \kappa=1) = 1$ 对所有 $|w|$ 成立：良态矩阵使用均匀权重
- $c(w; \kappa) \geq 1$——重要性不低于均匀情形
- $c(w; \kappa)$ 关于 $|w|$ 单调递增——大权重获得更高重要性
- $\dfrac{c(w_{\max}; \kappa)}{c(w_{\min}; \kappa)} \approx 1 + \alpha(\kappa - 1)$——动态范围由 $\kappa$ 决定

**定理（κ 加权 Lloyd-Max 最优性）。** 最小化 $\mathcal{D}_\kappa$ 的量化器 $Q_\kappa$ 满足：

1. **最近邻条件：**
   $$R_i = \{w : |w - q_i| \leq |w - q_j| \text{ 对所有 } j \neq i\}$$
   权重函数 $c(w; \kappa)$ 不依赖于量化索引 $i$，因此决策边界仍为 Voronoi 区域。

2. **κ 加权质心条件：**
   $$q_i^* = \frac{\int_{R_i} w \cdot c(w; \kappa) \cdot p(w) \, dw}{\int_{R_i} c(w; \kappa) \cdot p(w) \, dw} = \frac{\mathbb{E}[w \cdot c(w; \kappa) \mid w \in R_i]}{\mathbb{E}[c(w; \kappa) \mid w \in R_i]}$$
   这是 $R_i$ 内 $w$ 的 $c(w; \kappa)$ 加权条件均值。

**证明。**

量化器将 $\mathbb{R}$ 划分为 $K$ 个决策区域 $R_i = [\theta_{i-1}, \theta_i)$，其中 $\theta_0 = -\infty$，$\theta_K = \infty$，$\theta_i = (q_i + q_{i+1})/2$（$i = 1, \ldots, K-1$）。κ 加权失真为：

$$
\mathcal{D}_\kappa(\{R_i\}, \{q_i\}) = \sum_{i=1}^K \int_{R_i} c(w; \kappa) \cdot (w - q_i)^2 \cdot p(w) \, dw
$$

**步骤 1（固定 $\{q_i\}$，优化 $\{R_i\}$）。** 对于固定的 $w$，最优量化索引 $i$ 最小化 $(w - q_i)^2$。由于 $c(w; \kappa) > 0$ 不依赖于 $i$，因子 $c(w; \kappa)$ 消去：

$$
\arg\min_i c(w; \kappa) \cdot (w - q_i)^2 = \arg\min_i (w - q_i)^2
$$

因此最近邻条件不变：$R_i = \{w : |w - q_i| \leq |w - q_j| \text{ 对所有 } j \neq i\}$。

**步骤 2（固定 $\{R_i\}$，优化 $\{q_i\}$）。** 对于给定的 $R_i$，子问题为：

$$
q_i^* = \arg\min_q \int_{R_i} c(w; \kappa) \cdot (w - q)^2 \cdot p(w) \, dw
$$

对 $F(q) = \int_{R_i} c(w; \kappa) \cdot (w - q)^2 \cdot p(w) \, dw$ 关于 $q$ 求导并令其为零：

$$
\frac{\partial F}{\partial q} = -2 \int_{R_i} c(w; \kappa) \cdot (w - q) \cdot p(w) \, dw = 0
$$

$$
\int_{R_i} c(w; \kappa) \cdot q \cdot p(w) \, dw = \int_{R_i} c(w; \kappa) \cdot w \cdot p(w) \, dw
$$

由于 $q$ 在 $R_i$ 内为常数：

$$
q_i^* = \frac{\int_{R_i} w \cdot c(w; \kappa) \cdot p(w) \, dw}{\int_{R_i} c(w; \kappa) \cdot p(w) \, dw}
$$

**经验实现（有限样本）。** 给定 $W$ 的 $n$ 个权重样本 $\{w_j\}_{j=1}^n$，迭代 $t$ 时的经验质心更新为：

$$
q_i^{(t+1)} = \frac{\sum_{j \in R_i^{(t)}} w_j \cdot c(w_j; \kappa)}{\sum_{j \in R_i^{(t)}} c(w_j; \kappa)}
$$

其中 $c(w_j; \kappa) = 1 + \alpha \cdot \max(0, \kappa - 1) \cdot |w_j| / \max_j |w_j|$，$R_i^{(t)} = \{j : |w_j - q_i^{(t)}| \leq |w_j - q_k^{(t)}| \text{ 对所有 } k\}$。

**收敛性。** 交替最小化（坐标下降）单调收敛到 $\mathcal{D}_\kappa$ 的局部极小值：

- 步骤 1 按构造非增地减少 $\mathcal{D}_\kappa$（固定 $\{q_i\}$ 下的全局最优）
- 步骤 2 计算固定 $\{R_i\}$ 时 $\mathcal{D}_\kappa$ 的精确极小值
- $\mathcal{D}_\kappa \geq 0$ 且可行集有限（$K$ 层级），因此单调收敛到稳定点得到保证

**特殊情形。**

- $\alpha = 0$ 或 $\kappa = 1$：$c(w; \kappa) \equiv 1$，退化为标准 Lloyd-Max
- $\alpha > 0, \kappa \gg 1$：网格点向大 $|w|$ 值偏移，为权重分布的尾部分配更多量化层级

### 2.7 策略之 条件数正则化方法

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

| 参数     | 取值                                                            |
| -------- | --------------------------------------------------------------- |
| 参数量   | ~164M                                                           |
| 层数     | 12（其中 8 层 sliding window，4 层 full attention）             |
| 隐藏维度 | 768，FFN 中间维度 3072                                          |
| 注意力头 | 12 query 头，3 KV 头（GQA 4:1）                                 |
| 归一化   | 输入端 RMSNorm + 注意力后 RMSNorm + Q/K RMSNorm                 |
| 位置编码 | RoPE                                                            |
| 词表     | BPE 32K，自训练                                                 |
| 训练数据 | 4.24B tokens，4 个 tier（C4、FineWeb-edu、Wikipedia、OpenOrca） |
| 训练步数 | 2000 步（batch=8, seq_len=512）                                 |
| 随机种子 | seed=42（torch.manual_seed）                                    |

每个 tier 的后 5% 写入独立的 `tierN_val.bin` 文件作为评估集，校准（GPTQ Hessian、Lloyd-Max 网格拟合）只读 `tierN_train.bin`，评估只读 `tierN_val.bin`，由 `get_dataloader(split='train'|'val')` 在 dataloader 层强制隔离。

两个 FP16 检查点为：

- **fp16_baseline**：标准交叉熵训练；
- **cond_regularized**：在 CE 损失上叠加 $\lambda\sum_\ell\log\kappa(W_\ell)$ 正则项（实现于 `src/analysis/condition.py`），训练目标在于产出量化更友好的权重。

### 3.2 量化方案

模拟量化全部以 FP32 实现（无硬件 FP4 执行路径）。涉及的 PTQ 方法：

| 方法                   | Code                                     | 简述                              |
| ---------------------- | ---------------------------------------- | --------------------------------- |
| RTN (round-to-nearest) | `src/quantization/fp_quantizer.py`     | 逐通道缩放后取最近格点            |
| GPTQ                   | `src/quantization/gptq.py`             | 列序贪心量化 + Hessian 加权列补偿 |
| Lloyd-Max              | `src/quantization/adaptive_grid.py`    | 每层自适应 16 层级（FP4）         |
| MXFP4                  | `src/quantization/fp4_grids.py`        | 块缩放（block_size=32）+ E2M1     |
| Hadamard 旋转          | `src/quantization/hadamard.py`         | Walsh-Hadamard 旋转后量化         |
| Outlier 旋转           | `src/quantization/outlier_rotation.py` | 离群感知缩放 + Hadamard           |

格式覆盖 FP8 E4M3 与 FP4 E2M1 两种。两个检查点 × 两种格式 × 四种主方法，共计 $16$ 配置横向比较；其余方法（Hadamard、outlier、MXFP4）作为对照单独列出。

### 3.3 实验指标

我们选取了如下核心指标：

1. 单矩阵相对误差均值

   $$
   \overline{\frac{\|\delta y\|}{\|y\|}} = \frac{1}{|\mathcal M|}\sum_{W\in\mathcal M}\frac{\|(\hat W - W)X\|_F}{\|WX\|_F},
   $$

   对权重矩阵均权平均。这是定理 1 左边量的直接对应：单层 Linear 输出处的相对误差。
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

   $$
   frac{\|\delta y\|}{\|y\|} = \frac{\|(W_q - W)x\|_F}{\|Wx\|_F},
   $$

   其中 $x$ 是上述捕获的激活，$W$ 是 FP16 权重，$W_q$ 是量化后权重。实现中，对RTN等无状态方法，直接调用 `quantizer.quantize()`来计算；对于GPTQ/Lloyd-Max这类原地修改的方法，先保存 FP16 权重副本，再调用 `quantizer.quantize()`，最后计算误差时用 FP16 权重副本与量化后权重做差。
3. 条件数计算。用 `torch.linalg.svdvals(W)` 精确 SVD 计算 $\kappa(W) = \sigma_{\max}/\sigma_{\min}$（`src/analysis/condition.py:20-40`）。本项目矩阵最大维度为 3072×768，精确 SVD 的开销远小于 100 步评估批次，无需幂迭代近似。
4. 测量粒度。 $\|\delta y\|/\|y\|$ 在每个 Linear 层的输出处直接采集（即 $y = Wx$ 之后、RMSNorm/激活/残差之前），保证粒度与单层线性映射一致，不使用 PPL 等下游混淆指标。

此外，在保证严谨性上，采用多种子重复实验、多重检验校正、bootstrap 置信区间等方法，确保实验结果的可靠性。我们也保证了校准/评估数据分离，避免 in-sample 偏差。

## 四、实验结果

在本章中，我们首先对前述的四个定理给出实证检验。

### 4.1 定理 1 在单矩阵粒度上的验证

**实验设置。** 对模型中全部权重矩阵逐一量化为 FP4 E2M1，测量该层 Linear 输出的相对误差 $\|\delta y\|/\|y\|$，并记录 $\kappa(W)$ 与 $\|\delta W\|/\|W\|$。采用三种子复测。

**主结果。** Pearson $r = -0.2258$，$p = 3.89\times 10^{-2}$。

| 检验                                    | 结果                   |
| --------------------------------------- | ---------------------- |
| Pearson$r(\kappa,\|\delta y\|/\|y\|)$ | $-0.2258$            |
| 原始$p$                               | $3.89\times 10^{-2}$ |
| Bonferroni 阈值$\alpha$               | $5.95\times 10^{-4}$ |
| Bootstrap 95% CI（10000 次）            | $[-0.341,\,-0.139]$  |

按 Bonferroni 校正后 $p > \alpha$，定理 1 预测的"$\kappa$ 越大、量化误差越大"在单矩阵粒度上**不成立**；更进一步，$r$ 的符号是负的，意味着 $\kappa$ 与误差呈弱反相关。

**逐种子一致性。** 三种子结果一致呈轻度负相关，排除随机性偶然。

| Seed | Pearson$r$ |
| ---- | ------------ |
| 42   | $-0.180$   |
| 123  | $-0.280$   |
| 456  | $-0.207$   |

**子组分析。** 值得注意的是，FFN 子组上甚至出现强负相关 $r = -0.74$（量化误差随 $\kappa$ 增大而**减小**），这与定理 1 的结果并不一致。

| 子组      | 矩阵数 | Pearson$r$ | $p$                 |
| --------- | ------ | ------------ | --------------------- |
| Attention | 48     | $-0.171$   | $0.246$             |
| FFN       | 36     | $-0.739$   | $2.6\times 10^{-7}$ |
| 全局      | 0      | —           | —                    |

**可能的解释。** 分析所有矩阵的 $\|\delta W\|/\|W\|$，几乎都集中在 $0.14$–$0.16$ 这条窄带里，跟 $\kappa(W)$ 几乎无关。无论是 $\kappa \approx 5$ 的良态矩阵，还是 $\kappa$ 上千的病态矩阵，量化误差占比都相近。

原因藏在 FP4 的格点结构里。FP4 E2M1 只有 16 个浮点格点，单位舍入 $u = 0.25$。量化后 $\|\delta W\|/\|W\|$ 的取值更多由“权重如何分布在格点间隙中”决定，而非矩阵的谱结构，因此量化误差被格点间距钉死在某一固定量级。定理 1 的上界在这里并不是很好的估计：右端项为 $\kappa(W)\cdot(\|\delta W\|/\|W\|)$，其中第二个因子在不同矩阵上几乎恒定，那么右边在跨矩阵尺度上的变化就完全由 $\kappa$ 主导。可以推知，在 FP4 强量化下，真正控制输出误差的根本不是 $\kappa$。

从紧致比的视角来看（$\tau = \frac{\|\delta y\|/\|y\|}{\kappa(W)\cdot\|\delta W\|/\|W\|}$，$\tau = 1$ 意味上界紧；$\tau \ll 1$ 意味上界宽松。实验结果表明，$\tau$ 的均值为 $0.056$，最大 $0.138$，最小接近 0，定理 1 的上界平均比实际误差大了约 18 倍。事实上，根本原因在于方向的不对齐：$x$ 和 $\delta W$ 都不在最坏方向上，$(W+\delta W)x$ 与 $Wx$ 的差异就远小于两个范数相乘这种悲观估计。

**逐分量条件数验证。** $\kappa(W)$ 的失败促使我们转向逐分量条件数 $\text{cond}_{cw}(W,x) = \||W||x|\|/\|Wx\|$。对同一批单矩阵量化实验，额外计算每个矩阵的 $\text{cond}_{cw}$ 并与输出误差做相关分析。

**主结果。** Pearson $r = 0.928$，$p = 8.0\times 10^{-113}$，决定系数 $r^2 = 0.861$。

| 指标 | 结果 |
| ---- | ---- |
| Pearson $r(\text{cond}_{cw},\|\delta y\|/\|y\|)$ | $0.928$ |
| $p$ | $8.0\times 10^{-113}$ |
| 决定系数 $r^2$ | $0.861$ |

$\text{cond}_{cw}(W,x)$ 解释了 86% 的逐矩阵输出误差方差，而 $\kappa(W)$ 仅解释 2.5%。子组上同样显著：Attention 子组 $r = 0.90$，FFN 子组 $r = 0.95$。

**上界松紧度对比。** 规范界中位数过估计 1,523×；逐分量界中位数过估计 39.6×，比规范界紧 38 倍。由此得出：Transformer 权重符号混合、幅值近对数正态分布，使得 $\||W||x|\|/(\|W\|\|x\|)$ 典型值在 0.05–0.15 之间，逐分量界先验即更紧。实验中的 38× 改善量级与此估计一致。

### 4.2 定理 2 RMSNorm 阻断效应验证

我们首先进行 12 层逐层衰减比的测量。在 RMSNorm 输入端注入由权重量化产生的扰动 $\delta_{pre}$，在 RMSNorm 输出端测量 $\delta_{post}$，记录比值 $\rho_\ell = \|\delta_{post}\|/\|\delta_{pre}\|$。结果如下表：

| Layer | 输入端 RMSNorm$\rho$ | 注意力后 RMSNorm$\rho$ | $\|\delta_\parallel\|$ | $\|\delta_\perp\|$ |
| ----- | ---------------------- | ------------------------ | ------------------------ | -------------------- |
| 0     | —                     | 1.0161                   | 0.0000                   | 0.0000               |
| 1     | 0.2899                 | 0.2644                   | 0.1311                   | 0.0103               |
| 2     | 0.2450                 | 0.1983                   | 0.0985                   | 0.0088               |
| 3     | 0.2037                 | 0.1834                   | 0.0727                   | 0.0078               |
| 4     | 0.1802                 | 0.1718                   | 0.0655                   | 0.0075               |
| 5     | 0.1721                 | 0.1571                   | 0.0609                   | 0.0072               |
| 6     | 0.1570                 | 0.1420                   | 0.0748                   | 0.0112               |
| 7     | 0.1383                 | 0.1321                   | 0.0649                   | 0.0106               |
| 8     | 0.1300                 | 0.1167                   | 0.0596                   | 0.0101               |
| 9     | 0.1135                 | 0.1089                   | 0.0501                   | 0.0093               |
| 10    | 0.1052                 | 0.1009                   | 0.0456                   | 0.0089               |
| 11    | 0.0985                 | 0.0915                   | 0.0418                   | 0.0085               |

**结果分析。**

平均输入端衰减比 $\bar\rho = 0.1667$。衰减随深度增强，从 Layer 1 的 0.2899 单调下降到 Layer 11 的 0.0985，显示出RMSNorm 的强阻断效应。同时，正交分量远小于平行分量。误差被系统性地投影至与信号方向同向的子空间，但是总幅值仍是可控的，与理论的预测一致。

### 4.3 定理 4 Lloyd-Max 优于 uniform E2M1

**实验设置。** 对 fp16_baseline 与 cond_regularized 两个检查点的所有权重矩阵：(a) 直接做 uniform E2M1 RTN 量化；(b) 用每层权重在训练集上的样本拟合 Lloyd-Max 16 层级网格，然后量化。两种方式的输入激活、测量协议、矩阵范围完全一致。

**主结果。**

| Checkpoint       | 方法         | Mean$\|\delta y\|/\|y\|$ | Total$\|\Delta WX\|/\|WX\|$ |
| ---------------- | ------------ | -------------------------- | ----------------------------- |
| fp16_baseline    | uniform E2M1 | 0.0809                     | 0.0786                        |
| fp16_baseline    | Lloyd-Max    | **0.0664**           | **0.0640**              |
|                  |              | $-18\%$                  | $-19\%$                     |
| cond_regularized | uniform E2M1 | 0.0835                     | 0.0820                        |
| cond_regularized | Lloyd-Max    | **0.0680**           | **0.0660**              |
|                  |              | $-18\%$                  | $-20\%$                     |

两种指标、两个检查点上 Lloyd-Max 一致地降低误差 `18%-20%`。

**机制分析。** 4.1节讨论，FP4 上 RTN 的 $\|\delta W\|/\|W\|$ 几乎被 $u=0.25$ 的间隔钉死；Lloyd-Max 通过把 16 个层级移动到经验权重分布的密集区，直接缩小了 $\|\delta W\|$ 本身——而非借助 $\kappa$ 通道。换言之，Lloyd-Max 的成功来自定理 4 的最优性条件在 FP4 这种格点稀疏、分布敏感的场景下天然适配。

## 五、工业方案基准测试

本章在统一平台上系统对比六种 PTQ 方法在 FP8 与 FP4 两种格式、两个检查点上的表现。所有方法在同一组单遍捕获的 FP16 激活上量化，使用同一组验证集激活做评估。

| Checkpoint       | Format | Method    | Mean$\|\delta y\|/\|y\|$ | Total$\|\Delta WX\|/\|WX\|$ |
| ---------------- | ------ | --------- | -------------------------- | ----------------------------- |
| cond_regularized | FP8    | rtn       | 0.014120                   | 0.013824                      |
|                  | FP8    | gptq      | 0.021010                   | 0.020699                      |
|                  | FP8    | hadamard  | 0.510644                   | 0.496478                      |
|                  | FP8    | outlier   | 1.012394                   | 1.011499                      |
|                  | FP4    | lloyd_max | 0.068038                   | 0.066016                      |
|                  | FP4    | mxfp4     | 0.073486                   | 0.072057                      |
|                  | FP4    | rtn       | 0.083455                   | 0.082045                      |
|                  | FP4    | gptq      | 0.120697                   | 0.123132                      |
| fp16_baseline    | FP8    | rtn       | 0.013672                   | 0.013245                      |
|                  | FP8    | gptq      | 0.020392                   | 0.019897                      |
|                  | FP8    | hadamard  | 0.512558                   | 0.497926                      |
|                  | FP8    | outlier   | 1.012846                   | 1.011939                      |
|                  | FP4    | lloyd_max | 0.066427                   | 0.063950                      |
|                  | FP4    | mxfp4     | 0.071446                   | 0.069607                      |
|                  | FP4    | rtn       | 0.080922                   | 0.078578                      |
|                  | FP4    | gptq      | 0.116946                   | 0.118046                      |

FP8 排名：rtn < gptq < hadamard < outlier

FP4 排名：lloyd_max < mxfp4 < rtn < gptq

值得注意的是，Mean 与 Total 两个指标对四个有效方法（RTN、GPTQ、Lloyd-Max、MXFP4）给出完全相同的方法排名。这也说明了我们的指标设计中，等权指标低估 GPTQ 的顾虑可以排除。Hadamard 与 outlier 在 FP4 上失效。这两种方法在 FP8 上虽不优，但仍能保持模型可用；在 FP4 上误差发散，原因是它们设计时假定 FP8 的指数范围充足，旋转后激活动态范围扩张超出 FP4 仅 4 个指数位所能覆盖的范围。

### GPTQ 与 RTN 的比较

| Checkpoint / Format    | RTN Mean | GPTQ Mean | $\Delta$     |
| ---------------------- | -------- | --------- | -------------- |
| fp16_baseline / FP8    | 0.01367  | 0.02039   | **+49%** |
| fp16_baseline / FP4    | 0.08092  | 0.11695   | **+45%** |
| cond_regularized / FP8 | 0.01412  | 0.02101   | **+49%** |
| cond_regularized / FP4 | 0.08346  | 0.12070   | **+45%** |

Total $\|\Delta WX\|/\|WX\|$ 的对比同样给出 +50% 量级的劣化。这与先前研究的结果有所不同，在本研究的场景下我们给出如下分析：

 GPTQ 把矩阵按列序贪心量化，每列量化后将剩余列做线性补偿以抵消已发生的舍入误差：

$$
\text{comp} = -\delta_j\cdot\frac{H_{j,\,j+1:}}{H_{j,j}},\quad H = XX^\top.
$$

补偿信号本身在传递过程中也受 FP8/FP4 格点限制——补偿量被舍入回格点上，残差累积；最后一列没有"剩余列"可补偿，必须独自吸收所有累积补偿误差。在小模型（164M）+ 强量化（FP4）的场景下，列序补偿引入的噪声超过了它能消除的原始舍入误差。GPTQ 在 LLaMA-7B 等大模型上有效的前提（列数充裕、激活长尾稀疏）在本设置下不成立。

### 条件数正则化的效果

固定方法对比两个检查点：

| Format / Method | fp16_baseline | cond_regularized | $\Delta$ |
| --------------- | ------------- | ---------------- | ---------- |
| FP8 / rtn       | 0.01367       | 0.01412          | +3.3%      |
| FP8 / gptq      | 0.02039       | 0.02101          | +3.0%      |
| FP4 / lloyd_max | 0.06643       | 0.06804          | +2.4%      |
| FP4 / rtn       | 0.08092       | 0.08346          | +3.1%      |
| FP4 / gptq      | 0.11695       | 0.12070          | +3.2%      |

条件数正则化的初衷是主动降低 $\kappa(W)$ 以让权重对量化更友好，但在所有方法上均使误差变差约 3%。在当前的实验环境下，条件数正则化并不是一个好的策略，因为瓶颈与格点间隔，而非奇异值分布相关。

综上，对于本实验中的 FP4 量化，Lloyd-Max 是最优选择。

## 六、误差传播追踪

为了在层内逐细粒度地追踪量化误差，每个 Transformer block 内布置 7 个采样点：

| 点 | 位置                                   |
| -- | -------------------------------------- |
| P0 | input_norm 之前                        |
| P1 | input_norm 之后                        |
| P2 | attention 模块输出之后（o_proj 之后）  |
| P3 | attention 残差加法之后（P0 + P2）      |
| P4 | post_attn_norm 之后                    |
| P5 | FFN gate/up 融合之后（down_proj 之前） |
| P6 | FFN 残差加法之后（P3 + down_proj）     |

#### Layer 0

| Source Matrix                  | Type   | P0       | P1       | P2       | P3       | P4       | P5       | P6       |
| ------------------------------ | ------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `L0.attention.q_proj.weight` | weight | 0.000000 | 0.000000 | 0.031687 | 0.031602 | 0.032429 | 0.030216 | 0.030155 |
| `L0.attention.k_proj.weight` | weight | 0.000000 | 0.000000 | 0.014890 | 0.014851 | 0.015346 | 0.011435 | 0.011635 |
| `L0.attention.v_proj.weight` | weight | 0.000000 | 0.000000 | 0.027315 | 0.027242 | 0.027199 | 0.013356 | 0.014602 |
| `L0.attention.o_proj.weight` | weight | 0.000000 | 0.000000 | 0.075349 | 0.075148 | 0.075389 | 0.021444 | 0.028210 |
| `L0.ffn.gate_proj.weight`    | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.028761 | 0.026745 |
| `L0.ffn.up_proj.weight`      | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.027226 | 0.025318 |
| `L0.ffn.down_proj.weight`    | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.074774 | 0.069533 |

#### Layer 5

| Source Matrix                  | Type   | P0       | P1       | P2       | P3       | P4       | P5       | P6       |
| ------------------------------ | ------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `L5.attention.q_proj.weight` | weight | 0.000000 | 0.000000 | 0.024572 | 0.006648 | 0.006899 | 0.007994 | 0.006387 |
| `L5.attention.k_proj.weight` | weight | 0.000000 | 0.000000 | 0.031022 | 0.008393 | 0.008889 | 0.009891 | 0.008090 |
| `L5.attention.v_proj.weight` | weight | 0.000000 | 0.000000 | 0.053686 | 0.014524 | 0.015022 | 0.012932 | 0.013868 |
| `L5.attention.o_proj.weight` | weight | 0.000000 | 0.000000 | 0.066852 | 0.018086 | 0.018820 | 0.011892 | 0.017192 |
| `L5.ffn.gate_proj.weight`    | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.080348 | 0.013604 |
| `L5.ffn.up_proj.weight`      | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.063873 | 0.010815 |
| `L5.ffn.down_proj.weight`    | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.117297 | 0.019860 |

#### Layer 11

| Source Matrix                   | Type   | P0       | P1       | P2       | P3       | P4       | P5       | P6       |
| ------------------------------- | ------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `L11.attention.q_proj.weight` | weight | 0.000000 | 0.000000 | 0.018374 | 0.002731 | 0.002806 | 0.003263 | 0.002542 |
| `L11.attention.k_proj.weight` | weight | 0.000000 | 0.000000 | 0.022136 | 0.003290 | 0.003467 | 0.004484 | 0.003099 |
| `L11.attention.v_proj.weight` | weight | 0.000000 | 0.000000 | 0.047466 | 0.007056 | 0.007330 | 0.004864 | 0.006420 |
| `L11.attention.o_proj.weight` | weight | 0.000000 | 0.000000 | 0.056825 | 0.008447 | 0.008795 | 0.003349 | 0.007577 |
| `L11.ffn.gate_proj.weight`    | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.041951 | 0.008144 |
| `L11.ffn.up_proj.weight`      | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.033158 | 0.006437 |
| `L11.ffn.down_proj.weight`    | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.077734 | 0.015091 |

可以发现，误差注入后被注意力后 RMSNorm 与 FFN 的归一化吸收。同时，深层注入误差比浅层小，这反映了RMSNorm 的作用，这进一步验证了定理 2。此外，从0/5/11 层的误差传播结果来看，FFN 的 down_proj 注入误差最大，可能源于 FP4 在 FFN 中维度扩张，导致误差放大。

## 七、结论与展望

### 7.1 从理论到实践

本实验中验证了定理 2 和定理 4，并发现定理 1 在实际的量化中可能并不完全适用。我们通过对各个量化策略的比较，得出 GPTQ 和条件数正则化方法的局限，并验证了 RTN 的稳定性，以及 Lloyd-Max 量化方法在这一场景下的良好效果。

我们从中归纳出在真实训练中，对实践的指导：

- FP8 场景：用 RTN，不要用 GPTQ。FP8 的量化误差本身已经很小，Hessian 加权补偿引入的扰动反而更大。
- FP4 场景：用 Lloyd-Max，不要用 GPTQ 或旋转类方法。
- 不要用条件数 $\kappa$ 做精度分配。在 RMSNorm Transformer 中，激活方向集中在高奇异值子空间，与 $\kappa$ 敏感的最小奇异向量几乎正交，导致 $\kappa$ 系统性地高估误差且无预测能力。按 $\kappa$ 排序分配精度可能把高位资源放错位置。
- RMSNorm 是良好的误差阻断器，不需要额外干预。

### 7.2 研究贡献

1. 理论贡献：修正了$\kappa$ 作为量化指导信号的适用域，在小规模模型的 FP4 量化下，做针对性的理论分析；将 RMSNorm 与 Transformer 的误差传播机制相结合，以细粒度剖析误差阻断机制。
2. 实验贡献：设计了双指标评估协议以互相验证；在统一平台上完成各种量化策略横向对比并分析误差传播。
3. 工程贡献：提供可复现的统一平台，包括模型、量化工具、分析工具、实验脚本。

### 7.3 局限

1. 模型规模受限。164M 参数远小于工业 LLM；GPTQ 在大模型上的表现可能不同。
2. 采用 FP32 模拟。无硬件 FP4 实测，无法评估硬件层面的舍入与累加器实现差异。
3. 量化策略有限。本实验仅比较了 GPTQ、RTN 和 Lloyd-Max，实际应用中可能需要更多策略。
4. 模态受限。本实验仅针对文本 Transformer，其他多模态/全模态模型可能需要不同策略。

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
