# 理论展示配套讲稿

建议按 01 → 02 → 03 → 04 的顺序讲述，约 4 分钟。图片是理论示意，不是实验结果图。

**项目定位可以这样说：**“我们借助形状空间的几何思想，把神经网络和多项式模型的响应表示为可比较的形状，再用形状差异指导经典实验设计向神经网络的迁移。” 当前代码实际计算的是 **Procrustes disparity，即最优对齐后的平方残差**。函数名虽然保留了 `kendall_shape_distance`，实现已经明确替换了原来的 Kendall 测地距离。

## 图 01：从预测模型到形状空间

**建议讲稿：**“直接比较两套模型参数没有共同坐标系，所以我们在相同输入位置观察它们的输出。把输入和输出拼接，就得到两个对应的响应点云。中心化去除平移，归一化去除整体尺度，再把只差正交变换的表示视为同一个形状。这样，我们比较的是模型在数据区域中的响应形状。”

给定同一批输入 $x_i\in\mathbb R^p$，令 $m=p+1$：

$$G_f=\begin{bmatrix}x_1^\top&f(x_1)\\\vdots&\vdots\\x_n^\top&f(x_n)\end{bmatrix}\in\mathbb R^{n\times m},\qquad H=I_n-\frac1n\mathbf1\mathbf1^\top.$$

$$Z_f=\frac{HG_f}{\|HG_f\|_F},\qquad \mathbf1^\top Z_f=0,\quad\|Z_f\|_F=1.$$

中心化、单位范数的矩阵位于预形状球面 $\mathcal S\cong S^{(n-1)m-1}$。用 $[Z_f]$ 表示等价形状；实际实现允许旋转和反射，可用 $[Z_f]=\{Z_fR:R\in O(m)\}$ 描述其对齐等价关系。通常 Kendall 形状空间采用 $SO(m)$ 的旋转商；含反射的商不能不加说明地与它混称。商空间也可能有退化、奇异层，球面图片只是便于展示的低维示意。[Geomstats 形状空间文档](https://geomstats.github.io/api/geomstats.geometry.html#geomstats.geometry.pre_shape.KendallShapeMetric)

**讲述边界：**样本行有一一对应关系，不会自由重新匹配。这里没有额外定义流形体积测度或积分权重；可以把等权输入采样理解为经验参考分布 $\mu_n=n^{-1}\sum_i\delta_{x_i}$。现有实现直接对有限点云做矩阵运算。

代码依据：[响应点云构造](/C:/Users/sheesh/Desktop/bridge_DL_poly/measure_morala.py:558)、[中心化和归一化说明](/C:/Users/sheesh/Desktop/bridge_DL_poly/measure_morala.py:548)。

## 图 02：预形状球面上的弧与弦

**建议讲稿：**“归一化后的模型表示受到单位范数约束，因此不能把所有距离都理解成原始空间的直线长度。在几何层面，球面上的路径和穿过球体的弦有不同含义。这解释了我们为什么从形状空间看待模型差异。实际计算采用下一页的 Procrustes 差异。”

对已最优正交对齐的单位预形状代表，记

$$A=Z_f,\quad B=Z_g,\quad R^*\in\arg\max_{R\in O(m)}\langle A,BR\rangle_F,\quad c=\langle A,BR^*\rangle_F.$$

则单位球面夹角与弦长分别为

$$\theta=\arccos(c),\qquad d_{\mathrm{chord}}=\|A-BR^*\|_F=\sqrt{2-2c}.$$

球面弧长为 $\theta$，可用于说明预形状几何以及对齐后的商距离思想。一般黎曼距离定义为

$$d_{\mathcal M}(a,b)=\inf_{\gamma(0)=a,\,\gamma(1)=b}\int_0^1\sqrt{g_{\gamma(t)}(\dot\gamma(t),\dot\gamma(t))}\,dt.$$

**讲述边界：**图中每个端点代表整个归一化点云，不是某个输入样本；弧也不是沿 NN 响应曲面移动的路径。当前实验没有求解这个路径优化，也没有用球面弧长作为最终统计量。[Geomstats 预形状与商度量文档](https://geomstats.github.io/api/geomstats.geometry.html#geomstats.geometry.pre_shape.KendallShapeMetric)

## 图 03：项目实际计算的 Procrustes 差异

**建议讲稿：**“计算分四步：中心化、整体归一化、最优正交对齐，再允许一个最优的整体尺度。最后把所有对应点的残差平方相加。数值越小，代表两个响应点云越容易通过这些变换重合。”

沿用上页的 $A,B$：

$$\boxed{\Delta_P(f,g)=\min_{s\ge0,\,R\in O(m)}\|A-sBR\|_F^2.}$$

这里 $R$ 包含旋转或反射；$s$ 是对单位化后的第二个配置再次优化的单一尺度。因此最终对齐结果 $sBR$ 不一定仍在单位球面上。该式对应代码调用 `scipy.spatial.procrustes` 返回的 `disparity`。[SciPy 官方定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html)

**可用于答疑的推导：**因 $\|A\|_F=\|B\|_F=1$，目标函数为 $1+s^2-2s\langle A,BR\rangle_F$。取最优相关 $c\ge0$ 后，$s^*=c$，于是

$$\Delta_P=1-c^2=\sin^2\theta.$$

这是上述优化式的代数推导；它说明 **disparity、球面弧长和单位球面的弦长是相关但不同的量**。尤其不要把 $\Delta_P$ 标成 $\theta$ 或 $\|A-BR^*\|_F^2$。平方差异也不应直接宣称满足普通距离的三角不等式。

**讲述边界：**形状接近不保证预测 MSE 小。平移、整体尺度和正交变换会消除某些预测差异，因此需要单独报告原坐标下的预测误差。此外，$[X,f(X)]$ 中 $p$ 个输入列与一个输出列一起参与 Frobenius 归一化；输入维度和各列尺度会影响数值。跨维度比较时不能只凭距离绝对值判断效果。

代码依据：[实现与旧名称说明](/C:/Users/sheesh/Desktop/bridge_DL_poly/measure_morala.py:514)、[返回 disparity](/C:/Users/sheesh/Desktop/bridge_DL_poly/measure_morala.py:554)。

## 图 04：距离指导 D-optimal 与随机采样的比例

**建议讲稿：**“先用少量 pilot 数据训练初始神经网络，再构造两个多项式视角：拟合 NN 输出的 FullPR，以及从 NN 权重导出的 Taylor-PR。它们与 NN 的形状越接近，我们就越信任多项式特征上的 D-optimal 选点；差异越大，就保留更多随机探索。每轮更新训练集和信息矩阵，重新训练 NN。”

当前 Grand 主实验先计算

$$D=\tfrac12\{\Delta_P(\mathrm{NN},\mathrm{FPR})+\Delta_P(\mathrm{NN},\mathrm{TPR})\}.$$

设 $Q_{0.10},Q_{0.90}$ 为该次脚本运行中各案例、数据种子、pilot 种子的初始距离分位点：

$$\widetilde D=\operatorname{clip}\!\left(\frac{D-Q_{0.10}}{Q_{0.90}-Q_{0.10}},0,1\right),\qquad w=0.30+0.65(1-\widetilde D).$$

因此 $w\in[0.30,0.95]$。对批大小 $b$，

$$b_D=\operatorname{round}(bw),\qquad b_R=b-b_D.$$

先选 $b_D$ 个 D-optimal 候选，再从其余候选中随机选 $b_R$ 个。Grand 实验 $b=500$、更新 6 轮；例如 $w=0.95$ 对应 475 个 D-optimal 点和 25 个随机点，$w=0.30$ 对应 150 个和 350 个。

令 $z(x)$ 为含截距的标准化 FullPR 设计特征，已选集合为 $S_t$：

$$M_t=\lambda I+\sum_{i\in S_t}z_i z_i^\top,\qquad \ell_t(x)=z(x)^\top M_t^{-1}z(x).$$

经典正则化 D-optimal 目标是增大 $\log\det M_t$；单点增益由矩阵行列式引理给出 $\log(1+\ell_t(x))$。当前代码一次计算该轮信息矩阵的杠杆分数，再按分数取一批点，是 **batch leverage 近似 D-optimal**，不是逐点更新或全局最优批设计。

**讲述边界：**$D$ 和 $w$ 只由初始 pilot 确定，之后固定不变；循环中更新的是选点集合、信息矩阵和 NN。不要把回路画成每轮重算距离。距离最多使用 256 个验证输入点，超过时才随机抽样；当前 $n=10000$、75% 训练池、5% pilot、25% pilot 验证拆分时约有 94 个验证点，因此通常低于上限。两模型比较始终使用相同输入行。

这些数值是 README 中 Grand 主实验的设置：程序通用默认值仍是 `distance-combine=min`、$w\in[0.20,0.90]$，不可与主实验的 `mean`、$[0.30,0.95]$ 混淆。分位点相等时，代码有极差归一化及中点回退逻辑。

代码依据：[pilot 与两种 PR 构造](/C:/Users/sheesh/Desktop/bridge_DL_poly/measure_weighted_sampling_experiment.py:228)、[距离点数上限](/C:/Users/sheesh/Desktop/bridge_DL_poly/measure_weighted_sampling_experiment.py:210)、[分位数与权重](/C:/Users/sheesh/Desktop/bridge_DL_poly/measure_weighted_sampling_experiment.py:300)、[混合批选点](/C:/Users/sheesh/Desktop/bridge_DL_poly/measure_weighted_sampling_experiment.py:386)、[固定权重的轮次循环](/C:/Users/sheesh/Desktop/bridge_DL_poly/measure_weighted_sampling_experiment.py:421)、[batch leverage 实现](/C:/Users/sheesh/Desktop/bridge_DL_poly/iterative_highdim_dopt_experiment.py:187)、[Grand 复现实验参数](/C:/Users/sheesh/Desktop/bridge_DL_poly/PROJECT_README.md:78)。

## 演讲时推荐的收束句

“几何距离提供了一种迁移依据：当 NN 与多项式响应形状较接近时，更积极采用多项式上的实验设计；当它们差异较大时，增加随机探索。我们用实验检验这种规则的收益，而不把形状接近当作性能保证。”
