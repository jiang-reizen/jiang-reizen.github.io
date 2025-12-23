---
title: 'VCI exam3 复习'
publishDate: 2025-12-20
updatedDate: 2025-12-29
description: 'nothing'
tags:
  - VCI
language: 'English'
heroImage: { src: './thumbnail.jpg', color: '#61af85ff' }
---

## background

课上说会考一些主观题

先过滤出一份提纲

## Animation 动画

Character Animation Methods

skeleton system

通过四元数表示旋转，Global and Local

关键帧，补间动画：对位置和角度差值

正向运动学 FK 逆向运动学 IK

IK 是一个 optimization 问题 $p = f(\theta)$

CCD IK：迭代，每次通过旋转当前关节，让 终点/目标点/目前关节 在一条直线上。
- 简单且快速
- 更多末端执行器运动
- 它可能收敛缓慢，甚至不收敛
- 跟踪连续目标可能不稳定

Jacobian IK：可以求 $d f = J d \theta$ 其中  $d f,J$ 已知，那么可以算 $d\theta$，不断迭代优化

FABR 的思想是向前与向后两次伸展，反复迭代，直到达到精度要求。
- 向后伸展：将末端移动到⽬标位置，然后沿着末端到前⼀个关节的连线，根据⻓度找到前⼀个关节的新位置，不断重复这个过程。
- 向前延伸：向后延伸计算出来的起点位置和真正的起点不重合，那么强⾏移动到真实起点位置，然后向前运动学计算出末端位置。

Rigging and Skinning 绑定和蒙皮

这里是怎么做的？ PLACEHOLDER

## Data-driven Character Animation

Motion Capture 动作捕捉

外骨骼，惯性动捕；光学动捕（三维重建？），多摄像头无标记动作捕捉，多摄像头无标记动作捕捉；稀疏传感器的运动估计

Motion Synthesis 运动合成

整个 pipeline 是怎么工作的？ PLACEHOLDER

混合帧以确保平滑过渡

Motion Transition：Facing Frame ？ PLACEHOLDER

丢给 AI 整理吧，有些乱

## visualization-volvis

Scalar Field Visualization -> Marhcing squares 类似 Marching cubes，等高线，线性插值

Vector Field 当方向线段

LIC PLACEHOLDER

Tensor Field glyphs PLACEHOLDER

Basic Ray-Casting Algorithm

...


### 三种关键曲线

流线 streamlines

固定某个时刻 t0，曲线上的切线方向与该点的速度向量方向一致

迹线 Pathlines

单个粒子在一段时间内随流体运动的轨迹

脉线 Streaklines

从固定位置连续注入粒子（染料），在某一时刻所有粒子构成的连线。

**Key**：可以这么理解 pathlines 和 streamlines：如果位置可以写成关于 t0, t1 的二元函数（t0 时刻从 p0 出发 t1 时刻到哪里），那么 pathlines 就是 t0=t 让 t1 做自变量的函数，streaklines 就是 t1=t 让 t0 做自变量的函数。

流带 (Stream Ribbon)：两条相邻流线之间的带状物。

如果带子扭曲了，说明流体在旋转（Vorticity）。（想象一下发廊门口旋转的柱子）

流管 (Stream Tube)：封闭的流线形成管状。

如果管子变细，说明流速变快（根据流量守恒）。

### LIC

LIC 随机一张白噪音权重

根据流线的方向，计算白噪音加权后的结果

在真实方向上有平滑作用，在垂直方向上依然杂乱

### Feature-based / Topological Methods

临界点：v=0的点，计算该点雅可比矩阵的特征值

对两个特征值进行判断：
- 源：两实部为正
- 汇：两实部为负
- 鞍点：两实部一正一负
- 中心/漩涡：特征值有虚部。纯虚数闭合圆环，复数螺旋

拓扑骨架 (Topological Skeleton)

通过连接所有的临界点和分界线（Separatrices），我们可以画出一张简化的拓扑图。这张图就像是流场的“地图”，告诉我们在哪里会有漩涡，哪里气流会分离。

### 信息可视化

空间属性: 没有天然的空间坐标。我们需要人为设计空间映射（Layout）。

目标: 展现离散结构（Discrete Structures）和多维关系。

可视化设计的核心就是：如何将数据属性映射为**视觉通道**。

### 通道的有效性排名 (Ranking of Effectiveness)

1. 最精确 (Best): 位置 (Position) > 长度 (Length)。
   这就是为什么柱状图和散点图最容易读懂。
2. 中等: 角度/斜率 (Angle/Slope) > 面积 (Area)。
   这就是为什么饼图 (Pie Chart) 容易产生误读，因为人眼对面积和角度的判断不如长度准。
3. 最不精确 (Worst): 颜色饱和度 > 色相 > 体积。

人类对视觉通道的感知效率排序如下： 位置 > 长度 > 角度/斜率 > 面积 > 体积 > 颜色饱和度。 因此，在可视化设计中，应优先将关键数据维度映射到高感知效率的通道（例如用位置表示核心指标）。

### 感知原理 (Perception & Cognition)

- 前注意处理 (Preattentive Processing):利用颜色或形状突显异常值 (Outliers)。
- 格式塔原则 (Gestalt Principles):大脑倾向于将独立的元素视为一个整体。
  - 临近性 (Proximity): 离得近的物体被视为一组。
  - 相似性 (Similarity): 颜色/形状相同的物体被视为一组。
  - 连续性 (Continuity): 我们倾向于把断开的点看成连续的曲线。

### 可视化设计原则

Tufte 的极简主义原则：
- 最大化数据墨水比 (Data-Ink Ratio)
- 谎言因子 (Lie Factor)必须接近 1，图形中的变化幅度要和真实变化幅度一致

Shneiderman 的交互真言："Overview first, zoom and filter, then details-on-demand." “先**概览**，再**缩放过滤**，最后**按需查看细节**。”

### 高维

平行坐标：线条容易重叠混乱 (Visual Clutter)。解决方法是交互式高亮 (Brushing) 或半透明绘制 （PLACEHOLDER）

数据和平行坐标可视化的对应关系：
- 笛卡尔空间中的圆和椭圆映射成平行坐标空间中的双曲线；
- 笛卡尔空间中的旋转映射到平行坐标空间中的平移，反之亦然；
- 笛卡尔空间中的拐点映射为平行坐标空间中的顶点。


降维 (Dimensionality Reduction)：
- PCA (主成分分析): 线性降维，保留**方差最大**的方向
- MDS (多维尺度分析): 保持点与点之间的距离关系。核心思想是如果高维空间中的距离能够反映数据的相似性或差异性，那么在低维空间中保持这些距离关系，就能有效揭示数据的结构
- t-SNE / UMAP: 非线性降维，特别擅长把相似的数据聚在一起（Cluster Preserving），常用于机器学习可视化。（PLACEHOLDER）

图标法：
- 切尔诺夫脸谱 (Chernoff Faces): 用脸型、眼睛大小、嘴巴弧度分别映射不同的数据维度。（PLACEHOLDER）
- 星形图 (Star Plot / Radar Chart): 雷达图。

### 网格与图数据

力导向布局 (Force-Directed Layout)（原理: 物理模拟）

- 节点 (Nodes): 看作带电粒子，互相排斥（防止重叠）。
- 边 (Edges): 看作弹簧，互相吸引（保持连接）。

减少节点重叠：力导向布局通过综合作用力（排斥力和吸引力），在优化图结构的同时，最大限度地减少节点的重叠，使得图的可视化更加清晰易读。系统达到能量平衡时，也就是布局稳定时。通常能展现出很好的对称性和聚类结构。

计算复杂度，局部最优解

社交网络，聚类更加明显。

聚类布局（Cluster Layout）：
- 初始状态：每个节点最初都被认为是一个独立的类，即图中的**每个节点**都代表一个**单独的社区**。
- 合并步骤：在每一步中，算法计算**每对**社区之间的“**能量**”，“能量”越小意味着节点或社区之前的相关性或亲密度越高，选择**合并能量最小的两个社区**。
- 结束条件：算法持续进行合并，直到**所有节点合并成一个单一的社区**（整个图作为一个类）。
- 输出结果：最终，算法输出一个**划分树**，通常为完全二叉树，每一层表示一次聚类操作，每层的划分反映了图结构的不同层次。这使得用户可以从整体到局部观察图中的聚类关系。 通过这种聚类算法，Vizster 得以自动检测社区结构，并允许用户通过拖动滑块来调节划分出“社区”的个数，**社区个数越少、单个社区越大，社区内部人员的关系越紧密**。

环状图（Circular graph）：用嵌套的圆形排布节点，适用于树状结构或层次结构的可视化。根节点通常位于中心，子节点按层次向外辐射。

环形布局（Circular Layout）：将图中的所有节点排布在一个圆周上，边以弧线或直线的形式连接节点，被放置在圆周内部。着色和聚类方法，减少杂乱的连线交叉，优化曲线曲率和形状，提高环形布局图的可读性。



### 文本可视化

- 词云 (Word Cloud) 词的大小 = 出现频率
  
  EdWordle [WCB+18] 将每个词汇看成一个二维平面内的矩形刚体，在刚体上施加相互之间的吸引力、向画布中心的吸引力、阻尼力，并利用基于冲量的方法解除碰撞，而后通过刚体仿真器对词云刚体系统进行仿真，从而获得更紧密的词云排布。

  在尽可能保证邻居关系的基础上，通过绕中心的旋转查找，找到被修改词条的目标位置，并形成符合用户需求的紧密排布。

- 主题河流图 (ThemeRiver): 展示话题随时间的演变。

TIARA（Text Insight via Automated Responsive Analytics）[WLS+10] 采用流的形式来可视化文本：给定一组文档，TIARA首先使用主题分析技术将文档汇总为一组主题，每个主题由一组关键字表示（一层流）。除了提取主题外，TIARA还派生出随时间变化的关键字序列来描述每个主题随时间的内容演变（流中随横轴变化的关键字排布），关键字的纵轴宽度（每处流的纵轴宽度）表示了它出现的频率。

横轴上用类似词云的技巧，用大小表示聚类中谁更关键；纵轴上用纵轴宽度衡量这个聚类出现的频率

### 可视化工具

。。。

### 交互式可视分析

Visual Analytics = Visualization + Automated Analysis + Human Interaction

探索（Exploration）、验证（Verification）和知识生成（Knowledge Generation）。
数据有计算机管理处理，人类通过计算机分析数据，完成上面三步。

鱼眼透镜 (Fish-eye Lens): 鼠标指向的地方放大变形，周围压缩。

双曲树 (Hyperbolic Tree): 用于树状结构，中心节点很大，边缘节点指数级变小。

### 可视分析系统

可视分析系统的设计遵循 “数据-模型-可视化”三元框架
- 数据层（Data Layer）：负责数据的采集、存储、清洗和预处理。
- 模型层（Model Layer）：通过统计与机器学习模型抽象数据特征。
- 可视化与交互层（Visualization & Interaction Layer）：将数据或模型结果映射为可交互的视觉表示。

六大关键步骤：
1. 提出可视分析任务
2. 构建可视分析模型
3. 设计可视化方法
4. 实现可视化视图
5. 完成可视分析原型系统
6. 进行可视分析评测

### 经典交互技术

WIMP 范式:
- Windows (窗口)
- Icons (图标)
- Menus (菜单)
- Pointer (指针/鼠标)

等张 vs. 等长 (Isotonic vs. Isometric)
- 等张设备 (Isotonic):设备随手移动，位置发生改变，阻力很小（恒定）。 鼠标
- 等长设备 (Isometric):设备基本不动，感知的是用户施加的力/压力，且通常有自动回中 (Self-centering) 机制。 小红点

慢动时低增益（准），快动时高增益（快）。结合了两者优点。

### 交互技术评估

交互系统的可用性
- 易学性 （learnability）：易学性指的是用户学习交互系统相关功能的难度。
- 可发现性 （discoverability）：自主发现
- 潜在错误 （error-proneness）：潜在错误指的是用户在该交互系统中期望进行错误操作的次数。潜在错误的多少与以下两个特性有关：
  - 准确性 （accuracy）：准确性指的是为了得到最终计算结果，用户期望进行的操作次数。操作次数越多，产生错误结果的机会也会越多。
  - 可靠性 （reliability）：可靠性指的是系统产生运算错误的概率。系统越不可靠，单次操作返回错误结果的概率越高。
- 有效性 （effectiveness）：有效性指的是交互系统进行计算的效率高低。
- 满意度 （satisfaction）：满意度指的是用户是愿意持续性地使用该交互系统的交互方式。

系统可用性量表：分为积极的和消极的两组

A/B 评测（Formal A/B Studies）是简单随机对照实验其中一种。
进行测试的时候，测试者可以选择同时向用户展示两者，也可以向用户随机的展示两种交互系统的其中一种。
通常需要添加置信区间。

雅各布·尼尔森（Jakob Nielsen）提出的十大可用性原则：（part）
- 识别胜过记忆（Recognition rather than recall）

### 菲茨定律 (Fitts' Law)

从A点移动并点击B点需要多长时间

$$
MT = a + b \cdot \log_{2} (1 + \frac D W)
$$

a, b 经验常数；D 是 A 到 B 的距离，W 是目标大小

困难指数 $ID = \log_2 (1 + \frac D W)$ 单位 bits

目标越远 (D变大)、目标越小 (W变小)，点击就越难，时间越长

### VR/AR 中

射线投射 (Ray Casting): 像激光笔一样指。问题是手抖会被放大（海森堡效应），很难选准远处的物体。

虚拟手 (Virtual Hand): 直接用手去抓。直观，但受到臂长的限制（够不着远处的）。

## Temp

四元数

龙格-库塔法 (Runge-Kutta, RK4)

https://zh.wikipedia.org/zh-cn/龙格-库塔法

雅可比矩阵

24.2.2. 可视化流程 