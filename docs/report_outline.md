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

- 说明BA的基本定义和思路，在项目中的使用细节，包含数学定义:
    - 说明优化变量：相机旋转、相机平移和 3D 点坐标。
    - 说明固定项：当前实验固定 VGGT 内参，避免内参尺度漂移。
    - 说明损失函数：Huber robust loss 下的多视角重投影误差。
    - 说明两阶段流程：先全量优化，再剔除离群观测后继续优化。
- BA运行后的初步结果：对3个场景的结果展示。
    - 三个场景的基本结果
    - 两阶段运行剔除的外点数
    - 统计图展示BA的运行效果

## 6. 3D GS


### 自实现 3DGS 与官方 3DGS

- 说明3dgs的基本原理和自实现方式。
- 说明自实现 3DGS 已能训练和导出 checkpoint，但效果明显弱于官方实现。
- 写入自实现结果：custom BA + self 3DGS 在 `scene` 上得到的结果指标。render图展示自实现 3DGS 弱结果。
- 说明失败原因候选：densification、尺度初始化、学习率调度和 renderer 参数仍不稳定。
- 说明最终报告策略：自实现 3DGS 展示工程实现和失败分析，受限于进一步的工程优化最终质量展示使用 official 3DGS。


### human场景下基本3D GS结果展示

Human 场景修复实验

- 基本流程：使用 mask 白底合成，按 mask 过滤初始化点。初期运行效果较差。后续提升了VGGSFM的tracks数量。
- 说明关键发现：human 提升的主因是高密度 VGGSfM tracks，而不是 mask 本身。但是受限于硬件显存没办法进一步提升获得很好的结果。
- 展示两个human场景下的运行结果数值和最终render出的结果对比。
    - 写入 `1-human` 高密度 BA：`[PLACEHOLDER: no-filter rerun]`。
    - 写入 `2-human` 高密度 BA：`[PLACEHOLDER: no-filter rerun]`。
    - 写入 `1-human` final render：`[PLACEHOLDER: no-filter rerun]`。
    - 写入 `2-human` final render：`[PLACEHOLDER: no-filter rerun]`。

### 3D GS不同效果对比 

使用scene场景作为主实验，进行多种对比实验。说明具体实验设置。
- VGGT raw原始导出和BA后的3D GS效果对比，展示参数对比，体现出BA对于3DGS的重要性。
- 使用random init 的点云，对比VGGT raw 和BA后的3Dgs效果，即使不用点云初始化，BA 相机仍然能提升 3DGS。说明相机视角准确性才更重要，效果和使用tracker点初始化效果差别不大。

展示主实验的metrics对比。展示 GT、raw render 和 BA render 的视觉对比。

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
