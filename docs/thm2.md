# RMSNorm 的误差阻断效应

RMSNorm 将输入向量按其均方根进行缩放。对于输入向量 $x \in \mathbb{R}^d$，其 RMS 值为：

$$
RMS(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

$$
\hat{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i, \quad i=1,\dots,d
$$

其中 $\gamma_i$ 是可学习的缩放参数，$\epsilon$ 用于确保不除零。下面的推导中，我们都按照向量形式：

$$
\hat{x} = \frac{x}{RMS(x)} \odot \gamma
$$

下设 $f(x) = \sqrt d \cdot \frac{x}{\|x\|}$，这里取L2范数。当存在一个微小扰动 $\delta x$ 时（即 $\|\delta x\| \ll \|x\|$），可以对它一阶泰勒展开：

$$
f(x + \delta x) \approx f(x) + J(x) \delta x
$$

其中 $J(x)$ 是 RMSNorm 的雅可比矩阵，

$$
J(y) = \frac{\partial}{\partial y} \left( \frac{\sqrt{d}}{\|y\|} y \right) = \sqrt{d} \left( \frac{1}{\|y\|} I - \frac{y y^T}{\|y\|^3} \right) = \frac{\sqrt{d}}{\|y\|} \left( I - \frac{y y^T}{\|y\|^2} \right)
$$

注意到，$I - \frac{y y^T}{\|y\|^2}$ 是一个投影矩阵，保留的是与 y 正交的部分，而舍弃了 y 方向的分量。

$$
\delta_{out} \approx J(y) \delta y = \frac{\sqrt d}{\|y\|} \left( I - \frac{y y^T}{\|y\|^2} \right) \delta y
$$

其误差与 $\delta y$ 的比值为 $\frac{\sqrt d}{\|y^⊥\|}$，上界为 $\frac{\sqrt d}{\|y\|}$。
即误差与输入向量的范数成正比。一般情况下扰动是随机的，因此会十分逼近上界。

这一结果说明，RMSNorm 可以有效地阻断误差的传播，从而提高模型的稳定性。
