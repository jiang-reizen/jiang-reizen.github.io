---
title: CV Review Part1
publishDate: 2025-12-09
updatedDate: 2025-12-09
description: 'No decription.'
tags:
  - CV
heroImage: { src: './header.jpg', color: '#648083ff' }
language: 'English'
---

## Introduction

~~Copied from slide~~

- Computer vision
  - Reconstructs meanings (3D, semantics, etc.) from images
  - Image processing; 3D reconstruction; Deep learning
- Computer vision is hard
  - Semantic GAP
  - Projection, viewpoint, deformation, occlusion, illumination, motion, local ambiguities, variations
- History of computer vision
  - 1960-1970: **Heuristics methods**;
  - 1970-1981: **Low-level vision**;
  - 1985-1988: **Neural networks**;
  - 1990-2000: **3D geometry**;
  - 2000-2010: **Feature engineering**;
  - 2010-now: **Image Formation**

## Image Formation

Reconstructs meanings (3D, semantics, etc.) from images.

相机成像：小孔成像。

如果没有小孔那么 "All scene points contribute to all sensor pixels." 会导致一片模糊。

一组平行线 -> vanishing point (不一定在一个面上)

平面 -> vanishing line (平面上不同族的平行线) (一族平行面)

vanishing line 和 高度的关系？

小孔对成像的影响

Lens 保留小孔的性质，又能增加光照进入

Gaussian Lens 焦距和z的倒数的关系

Depth of Field 对于固定的 focal length ...，一个范围  acceptably sharp

The aperture size affects DOF 通过光圈调整 DOF

Field of View (FoV) The field of view is the angular extent of the world observed by the camera.

相机 tensor 范围有限 -> 决定视角场范围

Lens Aberrations：Radial Distortion，Chromatic Aberration，Vignetting

Colors 人眼，RGB，相机 Filter Array

Gamma 矫正

## Image Processing

亮度 +- 对比度 */

Pixel Processing: Histogram Equalization？

Filter: blur, sharpening

Seperable Filtering 2D = 1D * 1D

Correlation & Convolution

Correlation <-> matrix multi

Bilateral Filter 综合考虑距离和颜色

Filter -> Template Matching

Joint Bilateral Filter: Flash/No-Flash Pair
用 Flash 的 f 计算权重，这样色彩更加真实
同时用 Flash 提取细节信息，补充到结果照片中

Gaussian Pyramid：先 Blur 再 Downsampling

Up Sampling: Interpolation
u: interpolation kernel

Cubic Interpolation: 分三段，其中两段连接处导数相等, 需要归一化

nearets neighbour, bilinear, bicubic

Bilinear 有两种，一种放在角落，一种放在像素中间，两种有区别

Joint Bilateral Upsampling？

Laplacian Pyramid：与 blur /2 *2 的差，就是高频细节

Difference of Gaussian(DoG) is an approx. of Laplacian of Gaussion(LoG)

可以用于 Image Blending，通过 mask 合并不同频段的特征

Forward Warping 把 f 的每个像素传递过去，可能用空洞
Inverse Warping 计算 g 的每个像素，通过 f 插值

Image Morphing 提取特征，构造三角片面，然后找对应，并融合。

## Feature Detection

Partial Derivatives

Forward differencing / Central diﬀerencing

Prewitt / Sobel Kernel

Noise Severely Affects Derivatives of an Image -> Smooth image + filter

example: Gaussian Derivative (d signal * gaussian)

Canny Edge Detector:

1. Preprocessing: Grayscale conversion + Gaussion blur
2. Use central diﬀerencing to compute gradient image (instead of first forward differencing), which is more accurate.
3. Calculate **gradient magnitude** (why? what to do? and then do what?)
4. Non-Maximum Suppression
5. Double threshold
   
   Weak edges are further classified as edge or not in the next step
6. Edge Tracking by Hysteresis
   
   Weak edges that are connected to strong edges will be actual/real edges.

   Weak edges that are not connected to strong edges will be removed.

Harris Corner Detector:

...

数学推导处矩阵 M the second moment matrix

特征向量... -> the corner strength function R

- Preprocessing: Convert the input image to grayscale and apply a Gaussian filter
- Gradient: Apply the Sobel operator to find the x and y gradient values
- Harris value: For each pixel p, consider a local window around it and compute the second moment matrix M and the corner strength function R
- Threshold R and non-maximum suppression

如果对于像素进行线性的处理...
corners are partially invariant to affine intensity changes.

Corners are **not equivariant** with scaling.

Blobs are regions in a digital image that differ in properties, such as brightness or color, compared to surrounding regions.

一堆数学推导，之后看。

SIFT feature

blobs are scale-equivariant

scale normalization -> scale-invariant

Within the region around each keypoint, compute the gradient magnitude m(x, y) and orientation θ(x, y). Then construct a histogram of gradient orientations.

use Guassian weight to compute histogram

Normalize the feature to unit length to reduce the effects of illuminations. 
In practice, use 4x4 local window with 8 orientation
bins in each, resulting in a dimension of 128

Attention: Pyramid! Efficient to compute (why and how?)

Histogram of Oriented Gradients (HOG) Feature

1. Resize the image to 128x64, and calculate the gradients at each pixel.
2. The **unsigned** gradients (magnitudes and angles) are divided into **8x8 cells**, and a 9-point histogram is calculated in each cell.
3. **Group** 2x2 cells with a stride of 1.
4. Normalize the histograms (robust to illumination and shadings)

HOG vs SIFT

## Image Stitching

Panorama:

1. Extract feature points
2. Feature matching
3. Solve transformations
4. Blend Images

用齐次坐标进行点坐标变换（平移 Translate，放缩 Scale，旋转 Rotation，拉伸 Shear） Homogeneous Coordinates

解方程 -> 最小化模长 -> t=(A^TA)^{-1}A^Tb

研究为什么最后一行是 [0, 0, 1] 以及在三维中为什么就是 8 自由度了（归一化）

解 Homographies 列方程组，不同的技巧

这里求完偏导之后，得到的结论是 h 是特征向量，且对应的特征值就是 E 的取值，所以 h 取最小特征值对应的特征向量。

求 A^TA 的特征值和特征向量，可以对 A 用 SVD，取 V 的最后一列即可。

上述为  algebraic error for optimization.

RANSAC: RANdom SAmple Consensus

随机几个点（minimum sample size），计算出参数；根据参数分类为 inlier 和 outlier；不断重复，记录 inlier 最多的情况。重复次数足够或者 inlier 足够，就结束。最后用最好的 inlier 估计参数。

Suppose the proportion of inliers is G and the model needs P pairs to fit, the probability that we have not picked a set of inliers after N iterations: $(1 - G^P) ^N$

If we want a failure probability of at most 𝑒, then $N > \frac{\log e}{\log (1 - G^P)}$.

Blending:

Poisson Image Editing

.. Formula

- Solve the homography
- Warp the source image to the reference image
- Warp the mask image to the reference image
- Run the Poisson Editing algorithm by keeping the gradient in the mask region

## Camera Calibration

Calibration 校准

本节是在讲述，如何从照片中获取相机的参数信息，包括外参和内参。

内参可以认为是之和相机有关，而外参还和世界坐标的构建有关。

Resolve Single-view Ambiguity:

- Shoot light (lasers, structured light, etc.) out of the sensors
- Stereo: use 2 calibrated cameras in different views and correspondences
- Multi-view geometry: move the camera and find correspondence to solve X
- Shape from Shading: fix the camera, and reconstruct geometry with photos taken under different shadings
- Learning from data: train a neural network to predict the 3D information

Extrinsic parameters: 外参，世界坐标到相机坐标

Intrinsic parameters: 内参，相机坐标到成像平面

Extrinsic parameters including the rotation and translation
Intrinsic parameters including the focal length and the principal point coordinate
• 𝛼: aspect ratio, equals 1 unless pixels are not square
• 𝑠: skew, equals 0 unless pixels are shaped like parallelograms
• (𝑐𝑥, 𝑐𝑦): principal point, equals (w/2,h/2) unless optical axis doesn’t intersect the image center

Projection 的作用是什么？

Camera Calibration: Given 𝑛 points with known 3D coordinates 𝑿𝑖 and known image projections 𝒙𝑖, estimate the camera parameters

估计参数依然使用上面的套路。

这里由于解出来的矩阵是多个矩阵的复合，所以还要 perform RQ decomposition

In practice, **non-linear** methods are preferred

用含有 K，R，t 的式子表示损失（距离平方的和），最小化损失。

Triangulation: Given projections of a 3D point in two or more images (with known camera matrices), find the coordinates of the point 知道相机参数和一个点在不同照片上的位置，还原点原本的位置。

1. Geometric Approach: Find shortest segment connecting the two viewing rays and let 𝑿 be the midpoint of that segment 找两根射线最近的距离的中点（多个点时不适用）
2. Nonlinear Approach: 最小化距离平方和
3. Linear Optimization: 写成矩阵乘法的形式，用上面特征值的套路计算结果

Camera calibration using vanishing points:

If **world coordinates** of reference 3D points are **not known**, in special cases, we may be able to use vanishing points

vanishing point 和方向 D 的关系

Consider a scene with three **orthogonal** vanishing directions.
We align the **world coordinate** system with these directions.

此时的 P 矩阵的列向量有特别的含义。以及需要更多的条件。

由于正交，所以对于每一对 vanishing point 都有一组限制，不过 The constraints are nonlinear, but it’s not hard to solve.

以及 At least two finite vanishing points are needed to solve for both 𝑓 and 𝑐𝑥, 𝑐𝑦

总之比较复杂，之后晚点再来看

1. Solve for intrinsic parameters (focal length, principal point) using three
orthogonal vanishing points
2. Get extrinsic parameters (rotation) directly from vanishing points once
calibration matrix is known

## Epipolar Geometry

Stereo: use 2 **calibrated** cameras in different views and correspondences

在上一节中，确定了相机的参数，如何利用两个校准的相机，以及拍摄到的图片，获取有用的信息。（在上一节的 Triangulation 中其实已经是使用了两个校准过的相机）

Use the setup of two cameras
- We can calibrate the two cameras.
- We can find constraints for easier correspondence and easier 3D reconstruction.

Definition:

- center
- baseline
- epipoles
- point and projected points
- epipolar lines (intersection of epipolar plane and image plane)
- epipolar plane

Case: General, Parallel, Motion Perpendicular

Epipolar Constraint:

If we know point x, we have known epipolar plane, and also epipolar line l' where x' must lie on.

Whenever two points 𝒙 and 𝒙′ lie on matching epipolar lines 𝒍 and 𝒍′, the
visual rays corresponding to them meet in space, i.e., 𝒙 and 𝒙′ could be
projections of the same 3D point 𝑿. 在匹配的两条 epipolar line 上，任取两个点都能找到 3D 中的唯一的交点，不过由于这两个点可能本身是两个 3D 点的投影，所以这个交点不一定有意义。

Remember: in general, two rays do not meet perfectly in space due to
noise and inaccurate calibration! 当然噪声可能导致并不完美的交点。

上面是几何的描述，下面用矩阵乘法和齐次坐标对这个事实进行数学上的分析。

Epipolar Constraint: Calibrated Case

Suppose camera intrinsic parameters are known, and the world coordinate system is set to that of the first camera. 即知道 K, K', R, t 然后进行计算。

R, t 做作用是世界坐标系 -> 相机坐标系；
K 的作用是相机坐标系 -> image plane

这里的 x, x', X 都是齐次坐标系，所以用 ≅ 表示相等，x 和 x' 都是相机坐标系中的，带 pix 下标的是在 image plane 上的。以及 x 和 x' 是 normalized image coordinates （还是齐次坐标）

关于坐标的计算，首先是设 X = (x, 1) 这里的小 x 可以认为是坐标的值，然后和 x, x’ 有变量重名，需要小心理解。

推导可知 𝒙′, 𝑹𝒙, and 𝒕 are linearly dependent （齐次坐标系） -> 𝒙′∙ 𝒕 × 𝑹𝒙 = 0

然后公式变化一下：

$$
x'\cdot[t\times (Rx)] = 0 \Rightarrow x'^T[t_{\times}]Rx = 0 \Rightarrow x'^TEx = 0
$$

这里的 $E$ 就是 Essential Matrix。

以及从上述公式中，可以得到两条 epipolar line 的方程。

分析 E 的秩和自由度：

𝑬 is singular with a rank of 2
- [𝒕×] has a rank of 2
- 𝑹 has a rank of 3

𝑬 has 5 degrees of freedom
- Translation: 3 degrees of freedom
- Rotation: 3 degrees of freedom
- Scaling ambiguity

上面 E 的计算依赖于知道 K 和 K' 但是如果 K 和 K' 未知，那么就要用 F 了。

𝒙𝒑𝒊𝒙𝒆𝒍′𝑻 𝑭 𝒙𝒑𝒊𝒙𝒆𝒍 = 0 其中  𝑭 = 𝑲′−𝑇𝑬𝑲−1 

F 就是 fundamental matrix. 这里用的坐标就是 image plane 上的坐标。

F 的分析如下：

𝑭 is singular with a rank of 2
- [𝒕×] has a rank of 2
- 𝑹 has a rank of 3
- 𝑲 is of full rank

𝑭 has 7 degrees of freedom
- Det(F) = 0
- Scaling ambiguity

关于如何计算 F 矩阵，还是上面的套路，用特征向量来做。

但是这样解出来不是 2-rank 的，还要再用一步 SVD 分解，舍弃最小的奇异值，这样就是 2-rank 的了。

上面的方法是 Eight-Point Algorithm，但是由于计算精度等问题，需要归一化，所以有了 Normalized Eight-Point Algorithm。

简单来说就是对于传入的一组坐标，x 和 x' 分别归一化（batch normal），记录下归一化用的矩阵（归一化可以用矩阵运算表示），最后再补回来就行了。if 𝑻 and 𝑻′ are the normalizing transformations in the two images, then the fundamental matrix in original coordinates is 𝑻′𝑇𝑭𝑻.  详细推导应该不难。

上面的是 algebraic 方法，还有 geometric 方法，最小化距离和。

𝐿 = σ𝑖 dist(𝒙𝑖, 𝑭𝒙𝑖)2 + dist(𝒙𝑖, 𝑭𝑇𝒙𝑖′)2

Decompose R, t from an Essential Matrix

如果知道 F, K, K' 那么就能算出 E ，然后就能算 R 和 t，就能进行 triangulation

一堆数学推导，回来看。

## Two-View Stereo

Input: a stereo pair with known camera matrices (**calibrated**)

Output: a **dense depth map**

Basic Stereo Matching **Algorithm**

For each pixel in the first image:
- Find the corresponding epipolar line in the right image
- Examine all pixels on the epipolar line and pick the best match
- Triangulate the matched points to get depth information
- 
A Simple **Stereo System** : parallel, same height, same focal length (why same focal length?) -> epipolar lines = corresponding scanlines

A General Stereo System: use the fundamental matrix to find the homographies to project each view onto a common plane parallel to the baseline (Stereo Rectification) 转化为上一种情况

Stereo Image Rectification:

- Compute 𝑅1 from the essential or fundamental matrix (这个是假定第一个的世界坐标和相机坐标一致，计算第二个相对第一个旋转和平移)
- Rotate the right camera by 𝑅1 to align the orientation of two cameras
- Rotate the two cameras by 𝑅𝑟𝑒𝑐𝑡 so that the image planes are parallel to the baseline, i.e., map the original epipoles to infinity
- Scale both images by H to reduce the distortion

如何计算 R rect

Calculating the Rectifying Rotation Matrix

先计算出极点 e1，然后这个就是要旋转到的第一个坐标轴，然后...(具体的看p9的公式理解)，目标就是要让 epipolar line 平行

Depth from Disparity

在 Simple Stereo System 中，根据视差计算深度。

一通数学推导... B 的值应该是从 Calibration 的 t 那里计算出来的

Local Stereo Matching

比较简单的 Matching 就是滑动窗口，然计算匹配值，选最大的，根据平移的关系，可以排除掉右边的一部分不需要匹配。（有不同的匹配值计算方法）

但是存在遮挡的问题，可能在右侧找不到匹配的点。（Half Occlusions）

还有一些可能可能导致失败的原因：Textureless regions, Repeated Patterns, Non-Lambertian surfaces, specularities

上述原因会发生的其中一个原因是 Local，可以通过 Global 改进

Global Stereo Matching

Need non-local constraints and optimize the matching globally.

Some basic ans naive constraints:

- Uniqueness: Each point in one image should match at most one point in the other image.
- Ordering: Corresponding points should appear in the same order.
- Smoothness: Neighboring points should have similar disparity values, i.e., the disparities should be smooth.

May have some corner cases...

Disparity Space Image: We rearrange the diagonal band of valid values into a rectangular array (in this case of size 64xN).

由于两张图片的视差不会太大，所以我们可以先定一个最大视差阈值。如果把在左右扫描线上的坐标当作新的两个坐标轴，那么有意义的 dp 范围就呈现为一条带状，然后通过平移(Shear)可以认为是一个矩形，在这个矩形上 dp。

左侧的图可能会看到一些右侧看不到的像素，右侧会看到左侧看不到的，这个就是 Occluded Pixels。

由于我们希望尽量多的匹配，所以在 dp 中对 Occluded Pixels 进行惩罚。

这个 dp 里的 e 是什么意思。。

Deep Learning for Stereo Matching

Active Stereo with Structured Light

Project “structured” light patterns onto the object
- Simplifies the correspondence problem
- Use one camera and one projector

## Structure from Motion

- Input: many images captured by unknown cameras
- Output: camera parameters (motion) and a 3D model of the scene (structure)

和 Calibration Triangulation 的区别

要确定每个相机的 K R t 和 3d 空间中点的坐标 X

...

## Multi-View Stereo

- Input: several images of the same object or scene with **calibrated** cameras
- Output: compute a **representation** of the corresponding 3D shape

更加详细的来说 Input: several images of the same object or scene with calibrated cameras
- Arbitrary number of images (from two to thousands)
- Arbitrary camera **positions** (camera network or video)
- Often calibrate cameras with **structure from motion** or special devices 通过 SfM 校准相机

Basic Idea: Dense Correspondence across Images 一种简单的理解：
- Dense Correspondence 我们可以根据图像 1 和图像 2 构建深度-误差曲线，选取误差最小的作为最优深度
- across Images 多视角验证，3D 点投影到各个 照片上，比较各视角中 patch 是否相近

Visual Hull 三维凸包（一堆锥体的交）

Plane-Sweep Stereo

- For each depth plane
  - Compute homographies projecting each image onto that depth plane
  - For each pixel in the composite image stack, compute the variance
- For each pixel, select the depth that gives the lowest variance

原理：选择了一个深度后，如果某个点真的在这个深度上，那么投影后会重叠。下面用方差来衡量重叠程度（但是如果凑巧重合怎么办呢？）

对于每个深度的平面，把图像投影到上面去，对于每个像素可以计算这个深度上各个图像投影到的颜色的方差。

然后每个像素取所有深度平面中，方差最小的那个。

Depth Map based MVS

- Compute depth-maps using neighboring views
- Merge depth-maps into single volume
- Extract 3d surface from this volume

首先可以通过双目视觉计算相邻相机能够看到的深度图，然后合并。

为了效果更佳平滑，可以采用局部正则化。但是要注意，如果在合并前进行局部正则化可能导致合并后的不平滑，所以一般是合并后再正则化。

Patch-based MVS 本节课大头

Key idea: **patch-based reconstruction** + **iterative expansion and filtering**

1. Feature detection;
2. Feature matching; 
3. Patch expansion and filtering

具体来说如下（基于 Gemini 给出的答案）：

1. 找出纹理丰富的区域/点进行匹配，找 corner 或者 blob，对应前面的方法
2. 将 2D 特征点转化为 3D 的面片 patch
   
   通过 triangulation 找的 3D 中的坐标（前面有说到的多图像确定坐标），一个 patch 不只是中心点 c 的坐标，还有法向量 n，以及参考图像 R（看这个面最清楚的图像，通常是视线与法向量夹角最小的那个相机）。

   Patch: Position (x, y, z); Normal (nx, ny, nz); Extent (radius) 一个空间中有法向的小圆片

   初始化的 c 和 n 可能不够准确，所以用光度一致性（Photometric Consistency）来优化。将微小的 3D 面片投影回所有能看到它的相机图像中，如果 c 和 n 正确那么纹理应该非常相似（使用 NCC - 归一化互相关 来衡量），否则微调 c 和 n 提高 NCC。

   这一步之后得到了稀疏的点云
3. 面片扩散与过滤，通过“生长”使得覆盖物体表面
  
   - Patch Expansion: 如果某个位置有一个面片，那么它的邻域很可能也有一个表面，而且法向量和深度应该是连续变化的。
      
     取出一个现有的面片 P，将其投影到它的参考图像中。查看该面片在图像上的邻域像素（Grid cells）。如果某个邻域像素还没有对应的面片覆盖，就尝试在这里生成一个新的面片。

     假设新面片的 n' 和 d' 与原面片 P 相近，如果 NCC 足够高，那么保留这个面片并加入队列扩展。

   - Patch Filtering (过滤/剔除): 扩散过程非常激进，可能会产生很多错误的、悬浮的或重叠的面片（Outliers），需要清理。有下面三个方法：
     
     可见性一致性（Visibility Consistency）： 如果一个面片声称自己在一个位置，但那个位置在某些相机的视线中明显被其他面片遮挡了，或者在某些相机中看起来完全不像（Photo-consistency，NCC 很低），就剔除它。

     邻域约束： 一个合法的面片周围应该有其他面片（表面是连续的）。如果一个面片孤零零地悬在空中（Outlier），剔除它。

     重叠处理： 如果在同一个微小空间位置生成了多个面片（例如扩散“撞车”了），保留 NCC 分数最高（最像真实表面）的那个，剔除其他的。

上面的描述有不详细的地方，具体如下：

1. 如何计算 NCC：
   
   面片 P 有参考相机 R 和邻域相机集合 S（也能看到 P 的相机 V_i）。

   在 R 的图像上，以投影的点为中心取一个 $\mu \times \mu$ 的网格，将每一个像素点 p_i 反投影到面片上，得到 Q_i。然后将 Q_i 投影到每个 V_j 上获取对应颜色。

   这样就得到了两个颜色向量（R 和某个 V_j），计算**去均值归一化互相关**（公式自查）。

   然后对所有的 V_i 的相关度加权平均，记为 Score。
2. 微调：
   
   c 沿着参考相机的视线方向，前后移动。

   n 绕轴微小旋转（在欧拉角下改变偏航角和俯仰角）。

   尝试提高 score

3. 如何通过原面片 P 生成新面片 P' 的 n' 和 d':

   首先假设 n' = n ，用相机和相邻像素的射线与 P 的交点作为 c'。

   基于上面的初值微调。
   
课件中也 Gemini 不同之处在于，没有 R 这个参考图像，直接用 S 中两两 NCC 的和衡量 Score，所以在这里提取颜色向量，就是直接取两个图像上的投影点邻域网格。

以及最后要 Verify patch，具体如下：

首先要 Update S，计算两两的 NCC，并计算每张图片与其他的 NCC 的和，以最大的为主图像。然后根据 threshold（比如 0.7）筛选图片。

如果筛选后 |S| >= 3 那么认为 accepted。

难绷，课件偏离挺大的，整理一下。

Patch p 包含如下内容：

中心 c(p) 法向量 n(p) 可见性集合 V(p) 大小 9*9 像素

1. Feature Detection: Corners/ Blobs
2. Feature Matching:
  
   - 初始 Patch: c(p) 通过 triangulation 计算，n(p) 假设与参考图像平行，V(p) 初始化为 triangulation 的两张图像。
   - 优化 (Refinement): 利用光度一致性 (Photo-consistency) 来调整 c(p) 和 n(p)（上面说的微调，用两两 NCC 的和作为 score）
   - 扩充可见性集合 (Update V(p)): 将 Patch 投影到其他相机，如果与参考图像的 NCC > 0.7 则加入 V(p)
   - 验证 (Verification): 如果 |V(p)| < 3 那么丢弃
3. Patch Expansion and Filtering:

   - Patch Expansion (扩散):
     
     寻找空邻域，Patch p 投影到图像上，将所在网格标记为 occupied，检查周围未占用的网格生成新 patch。n(q), V(q) 和 p 相同，c(q) 为相机中心-空网格中心射线与 p 的交点

     对 q 进行 Refinement 和 Verification

   - Patch Filtering:
     
     比较 V(p) 和 NCC 分数来过滤，如果 |V(p)|*N(p_1) < sum_{i > 1}N(p_i)，那么就抛弃

缺点：

Surfaces must be Lambertian and well-textured.

The running time is relatively slow.

Today:
- COLMAP MVS 与像素级视图选择
- Towards Internet-Scale MVS: 将图像集分成若干个簇（Clusters） 。并行处理每个图像簇，最后合并结果。
- 基于深度学习的 MVS (Deep Learning for MVS)
- 神经辐射场 (NeRF - Neural Radiance Field)

## Photometric Stereo

## Optical Flow

## Image Recognition

线性分类器 (Linear Classifier)：神经网络中最基础的构建单元（全连接层）

线性分类器无法解决非线性可分的情况（Hard Cases），例如 XOR 问题、环形分布的数据 。这就是为什么我们需要后续的多层神经网络（非线性）

- SVM: “及格万岁”。只要正确类别的分数比别的类高出1分，它就不再关心了
- Cross-Entropy Loss (交叉熵损失 / Softmax Loss)，希望正确类别的概率无限接近1

## Neural Network

线性不可分问题：很多数据分布是线性分类器无法处理的。例如，异或问题（XOR）或环形数据分布（如 Slide 2 展示的红蓝点分布），无法画一条直线将它们分开

传统方法：
- 坐标变换
- 手动提取特征，HoG + SVM

单层感知机 (Single-Layer Perceptron, SLP) 无法解决 XOR 问题，但是多层感知机 (Multilayer Perceptrons, MLP)可以解决。

一般形式为 $s = W_2\sigma(W_1x)$ 其中 $\sigma$ 是激活函数。

关键就在于激活函数。激活函数必须是**非线性**的，且应该是**可导**的。

定理：只要有至少一个隐藏层，MLP 就可以以任意精度逼近任何函数。

通过组合两个 Sigmoid 或 ReLU 函数，可以构建出一个“凸起”或“台阶”（Basis Function）。无数个这样的“台阶”或“凸起”组合起来（线性加权），就可以拟合任意复杂的曲线或函数形状。

三种理解神经网络的视角：
- 视觉视角：线性分类器学了很多模板，神经网络的第二层对第一层学到的模板进行重组
- 几何视角：神经网络通过扭曲空间，实现了**非线性决策边界**。隐藏层单元（Hidden Units）越多，网络能拟合的**决策边界**就越复杂（如从简单的曲线变成复杂的多边形区域）
- 特征变换视角：原始空间 -> **可学习的特征变换 (Learnable Feature Transform)** -> 线性分类器。用可学习的特征变换替换手工特征变换。

足够宽只需要一个隐藏层即可拟合任何函数，但是这样需要的神经元数量极大，所以改为增加神经网络深度。

## Back Propagation

Gradient Descent (梯度下降) $w = w - \alpha \nabla_{w}L$。

Learning Rate is a **Critical** Hyperparameter. Learning Rate **Schedule**:
- Stepwise Decay: Reduce by some factor at fixed iterations
- Cosine Decay: reduce the learning rate following $\alpha_t = \frac{1}{2}\alpha_0(1 + \cos (\frac{t\pi}{T}))$
- Warmup：在训练刚开始时先线性增加学习率，然后再进行衰减。这有助于训练初期的稳定性。

计算图 (Computational Graph)，正向传播，反向传播，自动微分。

在 pytorch 中实现了自动微分和计算图的保存。

每个节点有上游梯度，本地梯度和下游梯度，用前二者计算后者。

关于 Jacobian Matrix:

$x \in \mathrm{R}^n,y \in \mathrm{R}^m$ 则 $\frac{\partial y}{\partial x} \in \mathrm{R}^{n\times m},(\frac{\partial y}{\partial x})_{i,j} = \frac{\partial y_j}{\partial x_i}$。

矩阵乘法的反向传播：

$$
Y = XW\\
\frac{\partial L}{\partial W} = X^T\frac{\partial{L}}{\partial Y}
$$

## Optimization 

Stochastic Gradient Descent: SGD

Approximate the sum using a minibatch of examples 每次用一组小 batch 进行训练，不使用全部的训练数据

这里的超参数有 Number of steps; Batch size (need shuffle); Weight initialization; Learning rate / step size

但是 SGD 有问题：
- 病态曲率 (Pathological Curvature / Ravines)，SGD在陡峭方向距离振动，平缓方向移动缓慢，收敛速度极慢
- 局部极小值 (Local Minima)
- 鞍点 (Saddle Points)：在一个方向上是极小值，在另一个方向上是极大值

有一些改进的方法。

Momentum (动量法)：模拟惯性 Build up the velocity for loss decent as a running mean of gradients

$v_{t+1}=\rho v_{t} + \nabla f(x_t), x_{t+1}=x_{t}-\alpha v_{t+1}$

一般采用 $\rho = 0.9$。在震荡方向（陡峭方向）：正负梯度相互抵消，减弱震荡；在前进方向（平缓方向）：梯度方向一致，速度叠加，加速收敛。在局部最小值和鞍点，也不会为 0。

Nesterov Momentum

比普通动量多了一步“预判”，用 $x_t+\rho v_t$ 计算梯度（TensorFlow 形式）。

自适应学习率算法 (Adaptive Learning Rate Methods)：

- AdaGrad：累积历史梯度的平方和。梯度大的参数，分母大，学习率自动减小；梯度小的参数，学习率保持较大。但是这样多轮以后可能导致变化极小，
- RMSProp: “Leaky Adagrad”。在上一个的基础上，引入“衰减系数” (Decay Rate)，让历史梯度的影响随时间指数衰减（Leaky Cache）。
- Adam: RMSProp + Momentum。分子和分母都有衰减系数，分子采用上面的Momentum，考虑之前的动量的影响，分母和 使用 RMSProp 对平方和时间衰减的累加。以及 Bias Correction: Normalize the accumulated moments by 1/(1 − 𝛽^𝑡)，由于初始化为 0 所以初期有偏差，引入修正项，分子分母的 beta 不同修正量不同。
- AdamW：Adam + Decoupled Weight Decay（解耦的权重衰减），如果把正则化项放到梯度中计算会被 Adam 的自适应机制扭曲。所以采用解耦的方法，度更新走梯度的路，权重衰减走权重的路，互不干扰。$w_{t+1}=w_t−lr\cdot AdamUpdate(g_t)−lr\cdot \lambda w_t$。
- Muon: MomentUm Orthogonalized by Newton-Schulz，旨在解决 AdamW 在大规模模型训练中的局限性。AdamW 内存消耗大，要维护一阶矩和二阶矩，以及忽视了矩阵结构。Muon 使用 正交化更新 (Orthogonalized Update) 尝试让参数的更新步长（Update Step）保持正交性，以及使用 Newton-Schulz 迭代 (Newton-Schulz Iteration)。（不会。。。）

二阶优化 (Second-Order Optimization)
- Newton's Method (牛顿法)：利用 Hessian 矩阵（二阶导数矩阵）直接找到极值点。时间开销过大，无法接受。
- L-BFGS：一种拟牛顿法，在大批量（Full Batch）且确定性的训练中效果很好，但在随机（Mini-batch）训练中很难应用

其他方法：
- 正则化 (Regularization)：提升泛化能力
- Model Ensembles (模型集成)：训练多个独立的模型，测试时取平均值
- Dropout 在训练过程中，以概率 p 随机将神经元的激活值设为 0
- Data Augmentation (数据增强)

## Convolutional Networks I



## RNN and Transformer

Transformer /sqrt(d) 保方差

multihead 一个相似度浪费参数

q, k normalize (around 1 is good)

I-GPT Image

Vision Transformer (Patches)
ViT (An Image is Worth 16*16 Words)

多个像素用一个 token

ViT 没有平移等变性，且没有什么层级结构，完全依靠大规模数据。而 CNN 有平移等变性且有层级结构。

加上层次结构 Swin Transformer
画窗口，轮流画 shifted window attention，有信息交互

## Image Segmentation

Semantic Segmentation 语义分割: Label each pixel in the image with a category label
不区分不同的 instance，只区分 label

Object Segmentation 圈出包围盒

更精细 Instance Segmentation

Semantic Segmentation with Sliding Windows: 对于每个滑动窗口，预测这个窗口的类，结果 label 存放在中心像素。

各种缺点：由于是小的网络，所以降采样不能太高。以及这样的预测开销较大，所以参数量不能太高，能力不足。小窗口，分的不准。

又慢效果又不好。

Make a CNN **Fully Convolutional** to Speed Up

上面的方法，相邻的块有共享的计算。全部都是卷积的神经网络，对图像的大小不敏感。
The convolutional part can be applied to a larger image and is fast as the computation is shared.

FC 实际上可以看作是一个 CNN kernel 大小为 feature map 的大小。
以及提高 feature 数的 FC，可以认为是 1*1 的卷积。

先训练一个分类的神经网络，然后把其中的 FC 全部变形成 CNN。

最后接一个小图变大图，上采样。把前面带着语义信息的 feature map 也用上(Skip Connections)，分层次上采样。
可以拼起来也可以加起来。

Deconvolution 反卷积（Other names: Fractionally strided convolution, Transposed Convolution），把一个值扩散到周围？（待确认）

前向 convolution 和反向 deconvolution 对应，或者反过来。如果前向和反向的算子可以一一对应，那么可以实现自动微分？

上述实现的是 **FCN**
- **Encoder**: a pretrained CNN 预训练的（例如网上下载）
- Make the CNN fully convolutional
- **Decoder**: upsample and fuse 输出语义，轻量级，主要训练这部分

Semantic Segmentation with U-Net

能不能训练对称的 Encoder 和 Decoder？（可能预训练提取的 feature 不好用）

U-Net: a bunch of convolutional layers, with downsampling, upsampling, and skip connections inside the network!
结构对称，在对称的结构上有 skip connection

U-Net 还能和 text 配合，把 text 融合进来。

Transformers for Semantic Segmentation

类似 Swin Transformer 分层级，做 Encoder，然后拼一个 FCN 做 decoder

或者 Progressive upsampling with 4 deconvolutions with stride 2, no skip connections. 也足够好

神经网络（Encoder， 如 ViT）**足够大足够强**，也蕴含了位置信息，就不需要 skip connections 了。（Encoder 足够强，Decoder 可以弱一些）

Facebook 的工作 [https://aidemos.meta.com/segment-anything](Segment Anything)

**A heavyweight image encoder** + lightweight decoder

- FCN: A pretrained CNN + a decoder for upsampling + skip connections.
- U-Net: symmetric encoder and decoder architecture with skip connections.
- U-Net and FCN are widely used in tasks in a “pixel-in pixel-out” fashion.
- Latest research: Use vision transformers for semantic segmentation.

## Image Generation

图像分类/图像分割 Supervised Learning

Unsupervised Learning 只学习 x 的分布 P(x)，然后采样 -> 图像生成

Generative Model

Autoregressive Models

explicit function for p(x) = f(x,W) through MLE (optimize KL distance)

x consists of multiple subparts x = (x_1, .. , x_r)

概率的链式法则（条件概率不断拆），建 T 维的概率密度 -> 建 1 维的概率密度（用 T 次）

PixelCNN and PixelRNN 逐像素生成，用历史生成的像素来预测目前这个。 -> PixelTransformer?

image-gpt 自回归图像生成。

igpt 能否做成 patch gpt？ 目前有实现的，比较复杂

Variational Autoencoder (VAE)

VAE 和 GAN 是生成模型中最热门的两种。（某段时间）

高维压缩到低维，然后再还原回高维（重建），Encoder 和 Decoder 不能有任何连接，否则可以认为是作弊。

用 neural network/PCA 做 Encoder 和 Decoder，神经网络看起来更有结构性，重建效果更好。

隐空间 Latent Space
Autoencoder 不是 Generative Model 因为空间比较空旷，对于没覆盖的区域生成效果极差。
解决方法：不映射成点，而是映射成高斯分布，解决空白的问题。

但是这里面有随机采样，而这一步不可微分。

Reparameterization trick: z = \mu_x + \sigma_x * \delta 其中 \delta ~ N(0, 1) 这样还能反传到 \mu_x 和 \sigma_x 上。

但是越扰动效果越不稳定，最后会收敛到 \sigma_x  = 0，坍塌了。加上正则化不让坍塌，加上与 N(0, 1) 的 KL。（KL Loss）

VQ-VAE-2

Generative Adversarial Network (GAN)

随机数发生器 U -> G -> X 噪音到照片

随机采样，拉近随机概率密度函数 X（假照片） 和真实概率密度函数 X^g （真照片）的距离

X, X^g -> D -> True/False 训练 D 的判定能力（判定是否为真实照片）

Train G and D alternatively, until convergence. 用固定的 D 训练 G（生成能力），用固定的 G 训练 D（判定能力）。

把 MLP 替换为 CNN，成 DCGAN

Diffusion Models

GAN 太不稳定。

文章 DDPM

人为的破坏，加噪音直至变成高斯噪音，记录路径作为训练数据，如何从高斯噪音还原真实照片。

Train 时先采样一个没有噪音的照片，然后采样一个时间步长，产生一个高斯噪音。
预测加了多少噪音，梯度下降。

按照 ... 公式加噪音，信号衰减+小噪音，破坏的过程。
根据公式可以推出 x_t 和 x_0 的关系，可以证明不断加噪音可以变成高斯噪音。

如何 Sampling？（跳回去，给定 x_t 去掉一些噪音得到 x_{t-1}）

第一个分布及其复杂，但是第二个分布稍微好估计一些。
从 x_{t-1} 预测 x_{t} 与 x_{0} 无关，所以分子的第一项是高斯。
得到均值和方差，按照这个来进行采样即可。