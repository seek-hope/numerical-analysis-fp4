# 随机舍入的累积误差界

## 问题描述

假设我们要计算 $n$ 个数的累加和 $S_n = \sum_{i=1}^n x_i$。设 $s_k$ 为第 $k$ 步的计算结果，即 $s_k = r(s_{k-1} + x_k$)= (s_{k-1} + x_k)(1 + \delta_k)$，$r$ 表示舍入。

每一次加法的绝对误差 $\epsilon_k = (s_{k-1} + x_k)\delta_k$，其中 $\delta_k$ 的上界是机器精度 $\epsilon_m$。

累积绝对误差为

$$
E_n \approx \sum_{k=1}^n \epsilon_k = \sum_{k=1}^n (s_{k-1} + x_k)\delta_k
$$

## 确定性舍入

最坏情况下，会不断地向同一个方向舍入，且到最大程度，即 $\delta_k = \epsilon_m$。此时，$\|E_n\|_{\infty} = O(nu)$。

## 随机舍入

$\delta_k$ 是随机变量，且期望 $\mathbb{E}[\delta_k] = 0$。此时，$\mathbb{E}[\text{round}(x)] = x$。

下面考虑证明 $\|E_n\|_{\infty} = O(\sqrt{n})$。

假设不同加法步骤的舍入误差相互独立：$\epsilon_1,\dots,\epsilon_n$ 是独立的随机变量。由协方差为0，可得

$$
\mathbb{E}[\epsilon_i \epsilon_j] = 0
$$

因此，总误差的 $L_2$ 范数（期望平方根）可以由各步方差的和给出：

$$
\mathbb{E}[E_n^2] = Var\left( \sum_{k=1}^n \epsilon_k \right) = \sum_{k=1}^n Var(\epsilon_k)
$$

对于有界均值为零的相对误差随机变量 $\delta_k$，方差有限（依据Popoviciu 不等式）必然存在一个常数 c，使得：
$Var(\delta_k) \le c u^2$

代回绝对误差 $\epsilon_k = (s_{k-1} + x_k)\delta_k$ 中：

$$
Var(\epsilon_k) = (s_{k-1} + x_k)^2 Var(\delta_k) \le (s_{k-1} + x_k)^2 c u^2
$$

下面给出放大估计，任何一步的部分和绝对值都不会超过所有输入项的绝对值总和 $X_{sum}$：$(s_{k-1} + x_k)^2 \le X_{sum}^2$。

代回总方差公式，对两边开平方根，得到在$L^2$范数下的误差界：

$$
\|E_n\| = \sqrt{\mathbb{E}[E_n^2]} \le \sqrt{c}\,\sqrt{n}u X_{sum} = O(\sqrt{n}u)
$$
