# VGGT + BA + 3D Gaussian Splatting 大作业

基于 VGGT 初始化的自研 Bundle Adjustment 与 3D Gaussian Splatting 流水线。

## 项目结构

```
cv_final/
├── data/scene/images/         # 64 张办公室环拍图像
├── vggt/                      # FacebookResearch/VGGT (上游, 不修改)
├── scripts/
│   ├── vggt_export.py         # VGGT 推理 + 轨迹导出
│   └── vggt_export.sh         # Shell 封装
├── src/
│   ├── data/
│   │   ├── reconstruction.py  # 统一 Reconstruction 数据格式
│   │   └── colmap_io.py       # COLMAP <-> .npz 转换
│   ├── ba/
│   │   ├── utils.py           # 投影、旋转、误差指标
│   │   ├── problem.py         # BA 问题构建 (稀疏 Jacobian)
│   │   ├── optimize.py        # 两轮 SciPy least_squares 求解
│   │   ├── run.py             # CLI 入口: python -m src.ba.run
│   │   └── synthetic_test.py  # 合成数据测试
│   ├── gaussian/
│   │   ├── model.py           # GaussianModel (灵活参数化)
│   │   ├── renderer.py        # gsplat CUDA 光栅化封装
│   │   ├── trainer.py         # 训练循环 + 增密/剪枝
│   │   ├── viewer.py          # Viser 交互浏览器
│   │   └── train.py           # CLI 入口: python -m src.gaussian.train
│   └── tests/                 # 单元测试
├── docs/
│   ├── plan.md                # 原始实施计划
│   └── design.md              # 设计文档 (思路/Pipeline/实验)
├── packages/LightGlue/        # 特征匹配库
└── outputs/                   # 运行输出 (自动创建)
```

## 环境配置 (uv)

```bash
uv python install 3.11

# 2. 创建虚拟环境并同步所有依赖（自动读取 uv.lock）
cd ~/cv_final
uv sync --index-url https://download.pytorch.org/whl/cu128

# 3. 激活环境
source .venv/bin/activate

# 4. 安装 LightGlue（editable install，pyproject.toml 中已配置 path）
cd packages
git clone https://github.com/cvg/LightGlue.git
uv pip install -e packages/LightGlue

# 5. （可选）安装 LPIPS 评估指标
uv pip install lpips
```

## 快速开始

### Phase 1: VGGT 图像初步提取

```bash
# 方式一: Shell 脚本
bash scripts/vggt_export.sh

# 方式二: Python 脚本 (可自定义参数)
python scripts/vggt_export.py \
    --scene_dir data/scene \
    --output_dir outputs/vggt_raw \
    --max_query_pts 2048 \
    --query_frame_num 5 \
    --vis_thresh 0.2 \
    --max_reproj_error 0.0 \
    --min_visible_frames 3

# 输出:
#   outputs/vggt_raw/reconstruction.npz   (统一格式)
#   outputs/vggt_raw/sparse/              (COLMAP cameras/images/points3D.bin)
#   outputs/vggt_raw/points3d_dense.ply    (稠密点云可视化)
```

### Phase 2: 自研 Bundle Adjustment

```bash
python -m src.ba.run \
    --input outputs/vggt_raw/reconstruction.npz \
    --output outputs/ba_custom \
    --huber_delta 1.0 \
    --outlier_threshold 5.0

# 输出:
#   outputs/ba_custom/reconstruction.npz   (优化后)
#   outputs/ba_custom/sparse/             (优化后 COLMAP)
#   outputs/ba_custom/ba_stats.json       (优化统计)
```

### Phase 3: 自研 3D Gaussian Splatting

```bash
python -m src.gaussian.train \
    --reconstruction outputs/ba_custom/reconstruction.npz \
    --image_dir data/scene/images \
    --output outputs/gs_custom_ba \
    --n_iterations 10000 \
    --resolution 768 432 \
    --sh_degree 2

# 输出:
#   outputs/gs_custom_ba/checkpoints/     (模型权重)
#   outputs/gs_custom_ba/validation/      (验证渲染图)
#   outputs/gs_custom_ba/final.ply        (最终 PLY)
#   outputs/gs_custom_ba/metrics.json     (训练曲线)
```

## 实验对照

| 实验 | BA | 3DGS | 目的 |
|------|-----|------|------|
| A | 无 | 自研 | VGGT 原始结果基线 |
| B | 自研 | 自研 | **分析 BA 对高斯效果的影响** |
| C | PyCOLMAP | 自研 | 验证自研 BA 有效性 |
| D | 自研 | 官方 gsplat | 对比 3DGS 实现 |
| E | VGGT 改进(预留) | 自研 | 论文改进实验 |

详见 [`docs/design.md`](docs/design.md)。
