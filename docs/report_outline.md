# 大作业报告提纲

说明：本文档作为报告的思路提纲，实际上的报告细节需要根据代码和实验结果补充，展示图可能需要根据实验数据生成assets. 最终需要得到一个完整的，问题定义清晰，结果展示和结果分析详细的大作业报告文档。

## 1. 摘要

- 写明项目完成了 VGGT、BA、3DGS 和训练-free VGGT 改进方案的端到端系统。、
- 说明任务定义和主要pipeline内容和细节：无相机标定的多视角图像输入，完成 VGGT 初始重建、自实现 BA 优化、3DGS 实时渲染和 VGGT 改进实验。
- 写明 VGGT 改进方向：视频关键帧选择和 depth-camera 稠密点过滤，目前已实现 frame selection 和 filtered dense export，部分最终指标待补。

## 2. 任务要求与系统设计

- 说明输入限制：只使用多视角图像，不使用给定相机标定。
- 说明 VGGT 作用：估计相机内参、OpenCV camera-from-world 外参、稀疏 tracks 和初始点云。
- 说明 BA 作用：固定内参，联合优化相机外参和 3D 点，最小化鲁棒重投影误差。
- 说明 3DGS 作用：用优化后的相机和点云初始化高斯，训练可实时交互渲染的场景表示。
- 说明改进实验作用：在 frozen VGGT 上做训练-free 改进，评估能否提高重建质量或速度。

## 3. 统一重建表示与评价指标

- 不只是列数据字段，要说明统一 `Reconstruction` 的设计目的：让 VGGT、BA 和 3DGS 在同一套相机、点云和观测图定义下衔接，便于做公平实验比较。
- 说明三类核心内容：图像和相机、三维结构、多视角观测图。
- 保留字段表：`image_names`、`image_size_hw`、`intrinsics`、`extrinsics`、`points3d`、`points_rgb`、`obs_camera_id`、`obs_point_id`、`obs_xy`、`obs_conf`。
- 说明坐标约定：OpenCV camera-from-world，`X_cam = R X_world + t`。
- 增加重投影公式：`u_hat = pi(K, R X + t)`，`e = ||u_hat - u||_2`。
- 定义后文统一使用的 RMSE、Median、P90。说明 RMSE 对大误差敏感，Median 表示典型观测误差，P90 表示长尾离群误差。
- 强调这些指标是自监督几何一致性指标，不需要真实标定，但不能完全等同于最终 3DGS 视觉质量。

## 4. VGGT 初始重建

- 补充原始文献引用：Wang 等，*VGGT: Visual Geometry Grounded Transformer*, CVPR 2025。
- 简要讲解 VGGT 的原理和思路：多视角图像整体输入，Transformer 聚合跨视角特征，预测相机、深度、点图和 tracks，作为无标定重建初值。
- 说明项目中的使用方式：frozen VGGT 初始化，不训练或微调；得到相机和点云后再进行 track prediction、可见性过滤和重投影过滤。
- 先展示大作业要求的 `1-human`、`2-human` 和 `scene` 三组初始重建结果。展示图像数、tracking 设置、点数、观测数和 VGGT raw 重投影 RMSE / Median / P90，并引用第 3 节的指标定义。
- 补充 `scene_32` 显存负结果：从 64 帧抽到 32 帧后可以把 `MAX_QUERY_PTS` 从 512 提高到 768，但最终 3DGS 效果低于 64 帧主场景，说明更多 tracks 不能替代充分视角覆盖。

## 5. 自实现 Bundle Adjustment

- 从大作业角度说明 BA 的作用：VGGT 是前馈初始化，BA 用多视角投影约束修正相机外参和三维点，使几何更一致。
- 说明为什么自实现：展示优化变量、损失函数、稀疏结构和离群点处理，而不是只调用黑盒工具。
- 说明固定项：固定 VGGT 内参，只优化外参和三维点，避免无标定输入下内参和尺度共同漂移。
- 说明优化变量：每个自由相机使用 SO(3) 旋转向量和平移向量，每个三维点使用 xyz 坐标；固定前两个相机作为 gauge anchor。
- 写清公式：
    - `u_hat_k = pi(K_i, R_i X_j + t_i)`
    - `r_k = u_hat_k - u_k`
    - `min_theta sum_k rho_delta(||r_k(theta)||_2)`
    - Huber loss 的分段定义。
- 说明稀疏 Jacobian：每条观测只依赖一个相机和一个点，因此只产生 `2x6` 和 `2x3` 参数块。
- 说明两阶段逻辑：第一轮全部观测 Huber BA；按第一轮重投影误差剔除离群观测；第二轮在过滤后观测图上继续优化。
- BA运行结果表保留 placeholder，不展示优化耗时，只展示 RMSE/Median/P90 before-after 和移除外点数。

## 6. 3D GS


### 自实现 3DGS 与官方 3DGS

- 说明 3DGS 的基本形式化表达：高斯中心、协方差、透明度、方向相关颜色，alpha compositing 渲染公式。
- 说明自实现方式：从 `Reconstruction` 点云初始化中心、邻域尺度、单位四元数、opacity、SH DC 颜色；用 `gsplat` 可微 rasterizer 和 L1+SSIM loss 训练；实现 densification/pruning 和 SH degree schedule。
- 展示 self 3DGS 与 official 3DGS render comparison，占位。
- 说明自实现 3DGS 已能训练和导出 checkpoint，但效果可能明显弱于官方实现。
- 说明失败原因候选：densification/pruning/opacity reset、尺度初始化、学习率调度、renderer 参数、背景处理和颜色稳定性仍不成熟。
- 说明最终报告策略：自实现 3DGS 展示工程实现和失败分析，受限于进一步的工程优化最终质量展示使用 official 3DGS。


### human场景下基本3D GS结果展示

Human 场景修复实验

- 基本流程：使用 mask 白底合成，可选按 mask 过滤初始化点。
- 简单说明 mask 对结果影响不大，tracks 密度可能更关键。
- 展示两个 human 场景下的基本指标表和最终 render comparison，不保留 human metrics bar 图。
- 增加 tracks 密度对比实验占位：low/high tracks 对 BA RMSE 和 3DGS 指标的影响。

### 3D GS不同效果对比 

使用 `scene` 场景作为主实验。说明 `scene` 是办公室视频流输入，背景噪声、反光、遮挡和纹理变化更复杂，是难度最大的主实验。
- 第一组：直接 VGGT raw sparse 初始化 vs BA sparse 初始化，说明 BA 对 3DGS 的作用。
- 第二组：random point init 下 raw camera vs BA camera，分离相机质量和点云初始化质量的影响。
- 不保留重复 metrics 图表，使用表格展示 metrics。
- 补充 render 图占位符：比较 GT 和 BA 后 3DGS 的最终渲染效果；另可补 random-init render comparison。

## 7. VGGT 改进

- 说明改进定位：受限于硬件条件，不可能对VGGT进行训练和微调，只改进输入帧选择和 VGGT 输出点云构造，使用现成的一些pipeline来做一些优化，实现训练-free、可解释、无需额外标注的改进思路。
- 说明最终改进的实现概述，
    - 说明改进一：视频帧选择用清晰度、曝光、纹理、去重、VGGT feature centrality、pose smoothness 和 diversity 选择关键帧。替代原来的均匀帧选择。
    - 说明改进二：通过depth + camera unprojection 和 point map 的一致性检查。进行几何过滤。


### 改进实验 I1：视频关键帧选择

具体的改进详细的定义和方法，包含必要的公式讲解。

当前使用的实验设置，选择帧写入的统计信息。

展示最终的筛选效果，通过筛选后和VGGT运行结果和vggt_raw对比，最终运行3D GS的效果对比。

### 改进实验 I2：Depth-Camera 稠密点过滤

先通过VGGT设计结构中Pointmap和depth重投影的重复性引入，运行并展示消融实验：
- 写入 depth-only 消融：用 VGGT depth + camera unprojection 初始化，结果 TBD。
- 写入 pointmap-only 消融：用 VGGT direct world point map 初始化，结果 TBD。

提出稠密点过滤的实际思路，具体的改进详细的定义和方法，包含必要的公式讲解。

当前使用的实验设置，过滤配置。

展示过滤后的基本信息和3D GS重建后的效果

### Full Method效果

最终的改进1，2结合的效果。对比这些整体改进后的3D GS重建效果，实际得到结果后对比基于VGGT_raw的Baseline得到实验结论。

## 8. 实时交互渲染展示

编译使用官方的可用viewer展示最好的重建的可视化截图。

## 9. 分析和结论总结

- 说明 BA 有效原因：BA 直接提升多视角几何一致性，使 3DGS 相机和初始化点更稳定。
- 说明 random init 现象：BA 相机本身也有贡献，不只是点云初始化贡献。
- 说明 human 特殊性：前景小和绿色背景会扭曲指标，需要 mask-white 和固定 test split。
- 说明 `scene_32` 负结果：更多 tracks 不能替代充分视角覆盖。
- 说明 self 3DGS 弱点：工程链路打通但训练细节未达到官方实现质量。
- 说明 VGGT 改进风险：训练-free filtering 可能减少点数并影响覆盖，必须用 3DGS 指标最终验证。

- 写明端到端系统已经完成：VGGT 初始重建、自实现 BA、3DGS 训练评估和 viewer。
- 写明最重要结论：custom BA 在主场景上显著降低重投影误差并提升 3DGS 渲染质量。
- 写明主定量结论：`scene` 的 `[PLACEHOLDER: raw RMSE] -> [PLACEHOLDER: BA RMSE]`，`[PLACEHOLDER: raw PSNR] -> [PLACEHOLDER: BA PSNR]`。
- 写明 human 结论：高密度 tracks 加 mask-white split 可获得稳定 human 展示结果。
- 写明 VGGT 改进结论占位：根据 I1-I4 最终结果填写“有效提升”或“负结果但提供分析”。

## 10. 未来工作

先留空
