 # VGGT + BA + 3D Gaussian 大作业实施计划

  ## 总体方案

  以当前 64 张办公室环拍图像为唯一主场景，形成以下流水线：

  图像 → VGGT 相机/深度/点云/轨迹 → 自研 BA 或 PyCOLMAP BA → 自研 3DGS 或官方 gsplat → 定量评
  估 + 实时展示

  核心对照：

   实验    BA                3DGS                   目的
  ━━━━━━  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━
   A       无                自研                   VGGT 原始结果
  ──────  ────────────────  ─────────────────────  ──────────────────────────
   B       自研 BA           自研                   分析 BA 对高斯效果的影响
  ──────  ────────────────  ─────────────────────  ──────────────────────────
   C       PyCOLMAP BA       自研                   验证自研 BA 的有效性
  ──────  ────────────────  ─────────────────────  ──────────────────────────
   D       自研 BA           官方 gsplat            对比自研与成熟开源 3DGS
  ──────  ────────────────  ─────────────────────  ──────────────────────────
   E       预留 VGGT 改进    自研 BA + 自研 3DGS    后续论文改进实验

  VGGT 改进方向暂不锁定，但所有中间结果统一为相同数据接口，使后续方法只需替换初始化阶段。

  ## 实现设计

  ### 1. 统一数据与 VGGT 初始化

  在 scripts 中实现独立调用入口，不修改 vggt 上游代码。

  输出统一重建文件，包含：

  - 图像名称、宽高和训练/验证划分。
  - 固定内参矩阵 K。
  - OpenCV 约定的 T_cw 外参。
  - 初始 3D 点、RGB 和置信度。
  - BA 观测图：camera_id, point_id, observed_xy, confidence。
  - 可供其他仓库使用的 COLMAP cameras.bin/images.bin/points3D.bin。

  使用现有 VGGT/VGGSfM tracker 建立跨视图轨迹。过滤低置信度观测、少于 3 帧可见的点和初始重投影
  误差过大的观测。无轨迹的 VGGT 稠密点云仅用于可视化或补充高斯初始化，不能直接用于 BA。
  
  现有vggt.sh 可参考路径配置和可视化方式，最终交付一个实现上面功能的Python脚本和调用python脚本的新.sh文件

  ### 2. 自研 Bundle Adjustment

  采用标准稀疏重投影 BA：

  [
  \min_{{R_i,t_i,X_j}}
  \sum_{(i,j)\in O}
  \rho\left(|\pi(K_i,R_i,t_i,X_j)-u_{ij}|_2^2\right)
  ]

  实现约束：

  - 固定 VGGT 内参，只优化相机外参和 3D 点。
  - 相机旋转使用 SO(3) 旋转向量，平移使用三维向量。
  - 固定前两台相机作为 gauge anchor，消除全局坐标系和尺度自由度。
  - 使用 SciPy least_squares、稀疏 Jacobian 结构和 Huber loss。
  - 第一轮优化后删除高重投影误差观测，再进行第二轮优化。
  - 失败时保留输入重建并明确报告，不静默输出无效结果。

  输出：

  - 优化后的统一重建文件和 COLMAP 模型。
  - 优化前后 reprojection RMSE、median、P90。
  - 有效点数、观测数、异常点删除比例。
  - 相机旋转/平移改变量、运行时间和收敛曲线。

  python -m src.ba.run \
    --input outputs/vggt_raw/reconstruction.npz \
    --output outputs/ba_custom

  同一观测图转换为 PyCOLMAP reconstruction，运行官方 BA 得到 outputs/ba_pycolmap，确保对比使用
  相同初值和观测。

  ### 3. 自研 3D Gaussian 框架

  “自研”范围为自行实现高斯模型与训练系统，仅复用 gsplat
  (https://github.com/nerfstudio-project/gsplat) 的 CUDA 可微光栅化算子。

  每个高斯包含：

  - 三维中心 xyz。
  - 三轴对数尺度。
  - 单位四元数旋转。
  - opacity logit。
  - 球谐颜色，先使用 degree 0，训练稳定后逐级提升至 degree 2。

  初始化：

  - 中心和颜色来自 BA 前或 BA 后的点云。
  - 尺度来自近邻点距离。
  - 四元数初始化为单位旋转。
  - opacity 初始化为 0.1。
  - 初始点最多 100,000，训练期间最多 300,000 个高斯。

  训练：

  - 432p 原始分辨率，固定每 8 张取一张作为验证集，共 8 张验证图。
  - 损失使用 0.8 × L1 + 0.2 × (1-SSIM)。
  - Adam 参数组分别控制位置、尺度、旋转、颜色和 opacity。
  - 约 10,000 iterations。
  - 500–6,000 iterations 间每 200 步执行增密。
  - 根据位置梯度 clone/split；根据低 opacity、异常尺度和屏幕半径进行剪枝。
  - 定期保存 checkpoint、PLY、验证渲染和训练指标。

  python -m src.gaussian.train \
    --reconstruction outputs/ba_custom/reconstruction.npz \
    --output outputs/gs_custom_ba

  实现基于 Viser 的交互 viewer：浏览器相机变化时调用 gsplat rasterizer，支持自由旋转、缩放、分
  辨率切换、背景色和高斯数量显示，并统计平均 FPS。

  ### 4. 开源基线

  - BA 基线：当前环境中的 PyCOLMAP。
  - 3DGS 基线：官方 gsplat simple_trainer，固定使用稳定 tag v1.5.3，而不是跟随持续变化的
    main。

  - 两个 3DGS 实现使用相同 COLMAP 输入、训练/验证划分、图像分辨率和近似训练步数。
  - 原始 GraphDECO 实现不作为主基线，因为安装复杂、显存占用更高；如时间允许，只补充一次定性结
    果。

  gsplat 官方提供 COLMAP trainer、实时
  viewer，并声明相较原始实现有更低显存占用，适合作为本作业开源对照。官方仓库
  (https://github.com/nerfstudio-project/gsplat) | GraphDECO 原始实现
  (https://github.com/graphdeco-inria/gaussian-splatting)

  ## 测试与实验

  ### 自动测试

  - 投影函数与 OpenCV/PyCOLMAP 投影结果一致。
  - SO(3) 参数更新不会产生无效旋转。
  - 合成 BA 数据加入相机和点云扰动后，重投影误差明显下降。
  - 固定相机、不可见点、负深度点和孤立轨迹处理正确。
  - COLMAP 与内部格式往返转换后相机和点云保持一致。
  - 3DGS 在 2–4 张小图上能过拟合，损失持续下降。
  - 高斯增密/剪枝后参数和 optimizer state 尺寸一致。
  - viewer 可以加载 checkpoint 并渲染有效图像。

  ### 报告指标

  BA：

  - 重投影 RMSE、median、P90。
  - 有效点和观测数量。
  - 优化时间。
  - 相机轨迹优化前后可视化。

  3DGS：

  - 验证集 PSNR、SSIM、LPIPS。
  - 训练时间、峰值显存、最终高斯数量。
  - 768×432 交互渲染 FPS。
  - 固定视角的 VGGT 原始、BA 后和真实图像对比。
  - 漂浮点、边缘细节、椅背和显示器区域的局部放大图。

  BA 对 3DGS 的结论必须使用实验 A/B：保持高斯实现和训练配置完全相同，仅替换相机与点云初始化。

  推荐服务器环境：

  - 单张 24 GB NVIDIA GPU，RTX 3090/4090/A5000 级别。
  - Ubuntu 22.04，Python 3.10 或 3.11。
  - 32 GB 以上系统内存，至少 50 GB 可用磁盘。
  - 与服务器驱动匹配的 PyTorch/CUDA、gsplat 1.5.3、PyCOLMAP、SciPy、OpenCV、Viser、LPIPS。
  - 默认单卡训练，不引入多卡复杂度。
  - 当前本地环境 GPU 较弱，因此主要用于代码、CPU BA、结果整理；3DGS 训练和答辩前性能测试在远
    程 GPU 完成。

  时间安排：

  1. 6月20–22日：统一数据格式、VGGT 轨迹导出、COLMAP 转换。
  2. 6月23–25日：完成自研 BA、合成测试和 PyCOLMAP 对照。
  3. 6月26–28日：完成自研 3DGS、增密剪枝和 viewer。
  4. 6月29–30日：运行 gsplat 基线及完整消融。
  5. 7月1日：补充选定的 VGGT 论文改进实验。
  6. 7月2日：整理表格、视频、PPT，并准备离线 checkpoint 和演示录屏。
  7. 7月3–5日：答辩时优先使用预训练结果实时浏览，避免现场训练或依赖远程网络。

  ## 假设与验收标准

  - 当前办公室场景为主场景，不新增公开或自采数据。
  - VGGT 改进方法后续选择，但必须输出统一 reconstruction 接口。
  - 自研 BA 固定内参，优化绝大多数外参和全部有效 3D 点。
  - 自研 3DGS 不从零实现 CUDA rasterizer，其余核心训练逻辑位于 src。
  - BA 后重投影误差应低于 VGGT 初值。
  - 自研 3DGS 能在验证视角产生可识别结果并通过 Viser 实时交互。
  - 最终至少完成 A–D 四组实验，并能定量回答“BA 是否改善 3DGS”。