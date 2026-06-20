# VGGT + BA + 3D Gaussian 大作业设计文档

## 1. 整体思路

### 1.1 背景与目标

以 64 张办公室环拍图像为唯一主场景，构建完整的三维重建与新视角合成流水线：

**输入** → 64 张环拍图像（`data/scene/images/000001.jpg` ~ `000064.jpg`）

**输出** → 优化后的 3D 高斯泼溅模型，可在 Viser 中实时交互浏览

核心任务分为三个阶段：

| 阶段 | 内容 | 自研范围 |
|------|------|----------|
| Phase 1 | VGGT 初始化 + 统一数据格式 | VGGT 图像提取脚本、轨迹过滤、COLMAP 导出 |
| Phase 2 | 稀疏重投影 Bundle Adjustment | 完整自研 BA（SciPy least_squares） |
| Phase 3 | 3D Gaussian Splatting 训练 + Viewer | 自研高斯模型/训练系统，复用 gsplat CUDA 光栅化 |

### 1.2 核心对照实验

保持高斯实现和训练配置完全相同，仅替换相机与点云初始化，分析 BA 对 3DGS 效果的影响：

| 实验 | BA | 3DGS | 目的 |
|------|-----|------|------|
| A | 无 BA | 自研 3DGS | VGGT 原始结果基线 |
| B | 自研 BA | 自研 3DGS | **分析 BA 对高斯效果的影响** |
| C | PyCOLMAP BA | 自研 3DGS | 验证自研 BA 的有效性 |
| D | 自研 BA | 官方 gsplat | 对比自研与成熟开源 3DGS |
| E | VGGT 改进（预留） | 自研 BA + 3DGS | 后续论文改进实验 |

### 1.3 设计原则

- **不修改 VGGT 上游代码**：通过 `sys.path.insert(0, 'vggt')` 复用 VGGT 模块
- **统一数据接口**：所有阶段通过 `Reconstruction` 数据类和 `.npz` 文件交换数据
- **模块解耦**：VGGT 导出、BA 优化、3DGS 训练各自独立，可单独运行
- **COLMAP 兼容**：所有中间结果同时输出 COLMAP 格式（cameras.bin / images.bin / points3D.bin），便于与 PyCOLMAP / gsplat 官方工具链互操作

---

## 2. 整体 Pipeline

### 2.1 数据流

```
data/scene/images/*.jpg (64张)
         │
         ▼
┌────────────────────────────────────┐
│  Phase 1: scripts/vggt_export.py   │
│  ────────────────────────────────  │
│  VGGT 推理 (518×518)               │
│    ├─ 相机位姿 (pose_enc → R|t)    │
│    ├─ 深度图 + 置信度               │
│    └─ 深度反投影 → dense 3D 点     │
│  VGGSfM 轨迹预测 (1024×1024)       │
│    ├─ 关键点提取 (ALIKED+SP)       │
│    ├─ 多视图轨迹追踪               │
│    └─ 可见性 + 置信度过滤          │
│  ────────────────────────────────  │
│  输出:                             │
│    outputs/vggt_raw/               │
│      ├─ reconstruction.npz         │
│      ├─ sparse/ (COLMAP 模型)      │
│      └─ points3d_dense.ply         │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Phase 2: python -m src.ba.run     │
│  ────────────────────────────────  │
│  加载 reconstruction.npz           │
│  BAProblem 构建:                   │
│    ├─ 变量: 6×(S-2) 相机 + 3×P 点 │
│    ├─ 固定前2相机 (gauge anchor)   │
│    └─ 稀疏 Jacobian 结构           │
│  Round 1: SciPy least_squares      │
│    ├─ Huber loss (δ=1.0)           │
│    └─ TRF 求解器                   │
│  异常点剔除 (>5px)                 │
│  Round 2: 重求解                   │
│  ────────────────────────────────  │
│  输出:                             │
│    outputs/ba_custom/              │
│      ├─ reconstruction.npz         │
│      ├─ sparse/ (COLMAP 模型)      │
│      └─ ba_stats.json              │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Phase 3: python -m src.gaussian.train │
│  ────────────────────────────────  │
│  高斯初始化:                       │
│    ├─ 中心/颜色 → 点云             │
│    ├─ 尺度 → 最近邻距离             │
│    ├─ 四元数 → 单位旋转            │
│    └─ 不透明度 → 0.1               │
│  训练循环 (~10k iterations):       │
│    ├─ gsplat CUDA 光栅化渲染       │
│    ├─ 0.8×L1 + 0.2×(1-SSIM) loss  │
│    ├─ Adam 参数分组                │
│    ├─ 增密 (500-6000 iter)         │
│    │   ├─ 高梯度小高斯 → clone     │
│    │   └─ 高梯度大高斯 → split     │
│    └─ 剪枝 (低opacity/异常尺度)    │
│  SH 渐进升级 (0→1→2 degree)       │
│  定期验证 + checkpoint             │
│  ────────────────────────────────  │
│  输出:                             │
│    outputs/gs_custom_ba/           │
│      ├─ checkpoints/*.pt           │
│      ├─ validation/*.png           │
│      ├─ final.ply                  │
│      └─ metrics.json               │
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  python -m src.gaussian.viewer     │
│  ────────────────────────────────  │
│  Viser 交互浏览器:                 │
│    ├─ 自由旋转/缩放                │
│    ├─ 分辨率切换                   │
│    ├─ 背景色调节                   │
│    └─ FPS / 高斯数量显示           │
└────────────────────────────────────┘
```

### 2.2 统一数据格式 (Reconstruction)

```python
@dataclass
class Reconstruction:
    # 相机 (S = 图像数量)
    image_names:    np.ndarray  # (S,) str       图像文件名
    image_size_hw:  np.ndarray  # (S, 2) int32   [height, width]
    intrinsics:     np.ndarray  # (S, 3, 3)      K 矩阵
    extrinsics:     np.ndarray  # (S, 3, 4)      [R|t] OpenCV camera-from-world

    # 3D 点云 (P = 点数量)
    points3d:       np.ndarray  # (P, 3)         世界坐标
    points_rgb:     np.ndarray  # (P, 3) uint8   颜色 0-255
    points_conf:    np.ndarray  # (P,)           置信度

    # 观测图 (N = 观测数量)
    obs_camera_id:  np.ndarray  # (N,) int32     相机索引
    obs_point_id:   np.ndarray  # (N,) int32     点索引
    obs_xy:         np.ndarray  # (N, 2)         像素坐标
    obs_conf:       np.ndarray  # (N,)           观测置信度

    metadata:       dict        # 额外元数据
```

### 2.3 BA 算法设计

优化问题：

$$\min_{R_i, t_i, X_j} \sum_{(i,j) \in O} \rho\left(\| \pi(K_i, R_i, t_i, X_j) - u_{ij} \|_2^2\right)$$

其中 $\rho$ 为 Huber loss，$\pi$ 为标准针孔投影。

约束：

| 约束 | 实现 |
|------|------|
| 固定内参 | 只优化外参和 3D 点，K 保持不变 |
| 旋转参数化 | SO(3) 旋转向量 (3 参数)，通过 Rodrigues 公式转换 |
| Gauge anchor | 固定前 2 台相机，消除 7-DOF 规范自由度 |
| 稀疏 Jacobian | 每个观测只涉及 1 个相机 (6列) + 1 个点 (3列) = 9 非零列 |
| 非线性求解 | SciPy `least_squares` + TRF 求解器 + `x_scale='jac'` |
| 两轮优化 | R1: 全量 Huber → 剔除 >5px 异常值 → R2: 重求解 |

### 2.4 3DGS 模型设计

每个高斯的参数：

| 参数 | 维度 | 激活 | 学习率 |
|------|------|------|--------|
| xyz (位置) | (N, 3) | 恒等 | 1.6e-4 |
| log_scales (对数尺度) | (N, 3) | exp() | 5.0e-3 |
| quaternions (旋转) | (N, 4) i,j,k,r | normalize() | 1.0e-3 |
| log_opacities (不透明度) | (N,) | sigmoid() | 5.0e-2 |
| sh_coeffs (SH 颜色) | (N, 3, (D+1)²) | SH eval | 2.5e-3 |

初始化策略：
- 位置/颜色来自 BA 或 VGGT 点云
- 尺度来自最近邻距离（k=3 的 KD-tree）
- 旋转初始化为单位四元数 (0, 0, 0, 1)
- 不透明度初始化为 0.1

训练调度：

| 阶段 | 迭代范围 | 操作 |
|------|----------|------|
| 增密 | 500–6000 | 每 200 步 clone (小高斯) / split (大高斯) |
| SH 升级 | 每 1000 步 | 0 → 1 → 2 degree |
| 剪枝 | 与增密同步 | 移除低 opacity / 异常尺度高斯 |
| 不透明度重置 | 与增密同步 | 将极低 opacity 高斯重置为 0.01 |
| 最大高斯数 | 训练全程 | 限制 ≤ 300,000 |

---

## 3. 后续等待运行的实验

### 3.1 实验矩阵

所有实验使用相同的 64 张办公室环拍图像，训练/验证按每 8 张取 1 张划分为 56:8。

| 实验编号 | BA 来源 | 3DGS 实现 | 输入路径 | 输出路径 |
|----------|---------|-----------|----------|----------|
| **A** | 无 BA (VGGT 原始) | 自研 | `outputs/vggt_raw/reconstruction.npz` | `outputs/gs_raw` |
| **B** | 自研 BA | 自研 | `outputs/ba_custom/reconstruction.npz` | `outputs/gs_custom_ba` |
| **C** | PyCOLMAP BA | 自研 | `outputs/ba_pycolmap/reconstruction.npz` | `outputs/gs_pycolmap_ba` |
| **D** | 自研 BA | 官方 gsplat | `outputs/ba_custom/sparse/` (COLMAP) | `outputs/gs_gsplat_ba` |
| **E** | VGGT 改进 (预留) | 自研 | TBD | TBD |

### 3.2 实验 A: VGGT 原始基线

```bash
# Phase 1: VGGT 导出 (已完成)
bash scripts/vggt_export.sh

# Phase 3: 直接用 VGGT 原始结果训练 3DGS (无 BA)
python -m src.gaussian.train \
    --reconstruction outputs/vggt_raw/reconstruction.npz \
    --image_dir data/scene/images \
    --output outputs/gs_raw \
    --n_iterations 10000 \
    --resolution 768 432 \
    --sh_degree 2
```

目的：建立 VGGT 直接输出的 3DGS 基线，作为后续 BA 实验的对照。

### 3.3 实验 B: 自研 BA + 自研 3DGS

```bash
# Phase 2: 自研 BA 优化
python -m src.ba.run \
    --input outputs/vggt_raw/reconstruction.npz \
    --output outputs/ba_custom

# Phase 3: 用 BA 优化后的结果训练 3DGS
python -m src.gaussian.train \
    --reconstruction outputs/ba_custom/reconstruction.npz \
    --image_dir data/scene/images \
    --output outputs/gs_custom_ba \
    --n_iterations 10000 \
    --resolution 768 432 \
    --sh_degree 2
```

目的：**核心实验**，通过对比 A/B 分析 BA 对 3DGS 质量的影响。

### 3.4 实验 C: PyCOLMAP BA + 自研 3DGS

```bash
# 使用 PyCOLMAP 官方 BA
# 需要将 VGGT 原始 Reconstruction 转为 PyCOLMAP 格式，运行官方 BA
python scripts/vggt_export.py --scene_dir data/scene --output_dir outputs/vggt_raw

# 然后使用 PyCOLMAP 的 BA（可复用 VGGT demo_colmap.py --use_ba 或单独脚本）
# 输出: outputs/ba_pycolmap/reconstruction.npz

# 训练
python -m src.gaussian.train \
    --reconstruction outputs/ba_pycolmap/reconstruction.npz \
    --image_dir data/scene/images \
    --output outputs/gs_pycolmap_ba \
    --n_iterations 10000 \
    --resolution 768 432 \
    --sh_degree 2
```

目的：验证自研 BA 效果是否达到 PyCOLMAP BA 水平。

### 3.5 实验 D: 自研 BA + 官方 gsplat

```bash
# 使用自研 BA 输出的 COLMAP 模型作为 gsplat simple_trainer 的输入
# 官方 gsplat simple_trainer 接受 COLMAP sparse/ 目录
python -m gsplat.train \
    --data_dir outputs/ba_custom/sparse \
    --data_device cpu \
    --output_dir outputs/gs_gsplat_ba
```

目的：对比自研 3DGS 与成熟开源 gsplat 的效果差异。

### 3.6 实验 E: VGGT 改进（预留）

```bash
# 待定：在 VGGT 推理阶段加入改进（如更好的轨迹跟踪、联合优化等）
# 输出: outputs/vggt_improved/reconstruction.npz
# 然后走同样的 BA + 3DGS 流水线
```

### 3.7 评估指标

| 模块 | 指标 | 说明 |
|------|------|------|
| **BA** | RMSE / median / P90 | 重投影误差（像素） |
| | 相机旋转/平移变化量 | 优化前后位姿变化 |
| | 优化时间 / 异常点比例 | 效率和鲁棒性 |
| **3DGS** | PSNR / SSIM / LPIPS | 验证集新视角合成质量 |
| | 训练时间 / 峰值显存 / 高斯数量 | 资源消耗 |
| | 交互渲染 FPS (768×432) | 实时性 |
| **消融** | B vs A | BA 对 3DGS 的影响 |
| | C vs B | 自研 BA vs PyCOLMAP BA |
| | D vs B | 自研 3DGS vs 官方 gsplat |

### 3.8 待安装依赖

```bash
# gsplat CUDA 光栅化器（Phase 3 训练所需）
pip install gsplat

# LPIPS 感知损失评估指标
pip install lpips
```

### 3.9 运行环境要求

| 组件 | 最低要求 | 推荐 |
|------|----------|------|
| GPU | 8 GB VRAM | 24 GB (RTX 3090/4090) |
| 系统内存 | 32 GB | 64 GB |
| 磁盘 | 50 GB | 100 GB |
| Python | 3.10 / 3.11 | 3.11 |
| CUDA | 11.8+ | 12.x |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
