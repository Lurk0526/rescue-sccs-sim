# SCCS: State-aware Clustering Search for Rescue Robots

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-brightgreen)](https://docs.ros.org/en/humble/)

**SCCS（State-aware Clustering Search）** 是一套面向复杂灾害环境的**多机器人协同救援仿真系统**。系统核心创新在于：

- 🎯 **ADC 自适应密度聚类** — 根据幸存者密度和机器人数量动态调整搜索区域划分
- 🏷️ **TCFM 三色荧光标记** — 通过环境物理标记（黄/绿/红）实现 O(1) 通信，消除显式通信瓶颈
- 🔬 **三级抗干扰感知流水线** — AIE 图像增强 → EKF 时空校验 → LiDAR 交叉验证

> 📄 论文：*复杂环境救援机器人聚类搜索方法*（Complex Environment Rescue Robot Clustering Search Method）

---

## 目录

- [系统架构](#系统架构)
- [核心算法](#核心算法)
- [项目结构](#项目结构)
- [实验设计](#实验设计)
- [实验结果](#实验结果)
- [快速开始](#快速开始)
- [ROS2 分布式部署](#ros2-分布式部署)
- [引用](#引用)
- [贡献者](#贡献者)

---

## 系统架构

```
┌──────────────────────────────────────────────────┐
│                  感知层 (Perception)               │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │   AIE   │→ │EKF Tracker│→ │LiDAR Validator  │  │
│  │图像增强  │  │时空校验   │  │强度交叉验证      │  │
│  └─────────┘  └──────────┘  └─────────────────┘  │
├──────────────────────────────────────────────────┤
│                  决策层 (Decision)                 │
│  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ ADC 聚类模块  │  │  TCFM 三色状态机           │ │
│  │ 自适应密度聚类 │  │  黄=待救 绿=已救 红=死亡   │ │
│  └──────────────┘  └────────────────────────────┘ │
├──────────────────────────────────────────────────┤
│                  执行层 (Execution)                │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ A* 寻路   │  │ 防碰撞移动 │  │ 物理标记喷涂   │  │
│  └──────────┘  └──────────┘  └────────────────┘  │
└──────────────────────────────────────────────────┘
                          ↕ (机器人 ↔ 环境 隐式交互)
┌──────────────────────────────────────────────────┐
│               RescueEnvironment                  │
│          80×80 栅格地图 | 30% 障碍 | 50 幸存者    │
└──────────────────────────────────────────────────┘
```

---

## 核心算法

### 1. ADC 自适应密度聚类

```
ε = 8.0 × (1 + 0.4·ln(K)) / (1 + 8·density)
```

- **K** = 机器人数量，**density** = 幸存者密度
- 密度越高 → ε 越小（精细划分）；机器人越多 → ε 越大
- DBSCAN 聚类后带负载均衡分配（score = distance + load × 3.0）

### 2. TCFM 三色荧光标记

| 颜色 | 状态 | 含义 |
|------|------|------|
| 🟡 黄 | 待救 | 已发现但未救援 |
| 🟢 绿 | 已救 | 救援完成 |
| 🔴 红 | 死亡 | 生命强度归零 |

**通信复杂度**：O(1)（显式通信 O(N²)），通过环境标记实现隐式协同。

### 3. 贝叶斯雷视融合

```
P_fused = (P_lidar × P_camera) / (P_lidar × P_camera + (1-P_lidar)(1-P_camera))
```

融合 LiDAR 高斯衰减检测和视觉概率检测的独立贝叶斯结果。

### 4. 三级抗干扰感知流水线

- **Level 1**: AIE 自适应图像增强（CLAHE + 暗通道先验去霾 + 伽马校正）
- **Level 2**: EKF 时空一致性校验（τ=3 帧确认机制 + 最近邻数据关联）
- **Level 3**: LiDAR 强度通道交叉验证（荧光材料反射率 0.82-1.0 vs 砖块 0.22-0.68）

---

## 项目结构

```
rescue_sim/
├── env.py                  # 救援环境模拟（地图/幸存者/生命衰减）
├── robot.py                # 机器人智能体（贝叶斯感知/决策/移动/救援）
├── tcfm.py                 # TCFM 三色荧光标记 + 广播通信对比
├── pathfinding.py          # A* 寻路算法（曼哈顿启发式）
├── run_experiments.py      # 一键实验运行（4 实验 + 6 图表）
├── test_perception.py      # 感知模块独立测试
├── docker-compose.yml      # ROS2 Docker 部署配置
│
├── algorithms/             # 核心算法
│   ├── sccs.py             # SCCS 系统（ADC + TCFM 集成）
│   └── baselines.py        # 对比算法（Greedy/PSO/Random）+ 消融变体
│
├── perception/             # 三级感知流水线
│   ├── aie.py              # AIE 自适应图像增强
│   ├── ekf_tracker.py      # EKF 时空一致性校验
│   ├── fusion_pipeline.py  # 雷视融合完整流水线
│   └── lidar_validator.py  # LiDAR 强度交叉验证
│
├── ros2_ws/                # ROS2 Humble 分布式仿真
│   └── src/rescue_sccs/
│       ├── package.xml
│       ├── setup.py
│       ├── launch/rescue.launch.py
│       └── rescue_sccs/
│           ├── env_node.py
│           ├── robot_node.py
│           └── sccs_node.py
│
├── output/                 # 实验结果
│   ├── data/               # CSV 实验数据
│   │   ├── sota_results.csv
│   │   ├── ablation_results.csv
│   │   ├── extreme_results.csv
│   │   └── scale_results.csv
│   └── figures/            # 论文级别图表（300 DPI）
│       ├── fig1_coverage_curves.png
│       ├── fig2_sota_boxplots.png
│       ├── fig3_ablation.png
│       ├── fig4_extreme_env.png
│       ├── fig5_scalability.png
│       └── fig6_comm_cost.png
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## 实验设计

### 仿真环境
| 参数 | 值 |
|------|-----|
| 地图尺寸 | 80×80 格（40m×40m） |
| 障碍物占比 | 30% |
| 幸存者数量 | 50 个（5 个高斯聚集簇） |
| 最大步数 | 4000 步 |
| 随机种子 | 30 个 |

### 实验组

| 实验 | 目的 | 对比方法 |
|------|------|----------|
| **Exp1** SOTA 对比 | 验证 SCCS 整体性能 | SCCS vs Greedy vs PSO vs Random |
| **Exp2** 消融实验 | 量化各模块贡献 | Full vs w/o TCFM vs w/o 聚类 vs w/o LiDAR |
| **Exp3** 极端环境 | 验证鲁棒性 | SCCS 融合 vs Greedy 纯视觉（正常/扬尘/弱光） |
| **Exp4** 可扩展性 | 验证规模化能力 | SCCS vs Greedy（N=3/5/7/9 台机器人） |

---

## 实验结果

### SOTA 对比（Exp1）

| 方法 | 覆盖率 | 完成步数 | 负载方差 | 通信开销 |
|------|--------|----------|----------|----------|
| **SCCS（本文）** | **0.911±0.118** | **193** | **9.89** | **0 条** |
| Greedy | 0.858±0.086 | 258 | 30.34 | 197 条 |
| PSO | 0.835±0.096 | 295 | 44.52 | 226 条 |
| Random | 0.569±0.170 | 488 | 30.88 | 0 条 |

- 覆盖率领先 Greedy **6.2%**，完成时间缩短 **25.2%**
- 负载方差降低 **67.4%**，通信从 197 条降至 **0 条**

### 消融实验（Exp2）

| 变体 | 覆盖率 | 完成步数 | 通信开销 | 贡献 |
|------|--------|----------|----------|------|
| Full SCCS | 0.911 | 193 | 0 条 | 基准 |
| w/o TCFM | 0.913 | 190 | 142 条 | TCFM 消除 100% 通信 |
| w/o 聚类 | 0.770 | 258 | 0 条 | 聚类提升覆盖率 18.3% |
| w/o LiDAR | 0.767 | 355 | 0 条 | 融合提升覆盖率 18.8% |

### 极端环境鲁棒性（Exp3）

| 环境 | SCCS 融合 | Greedy 纯视觉 | 差距 |
|------|-----------|---------------|------|
| 正常 | 0.911 | 0.732 | +17.9% |
| 扬尘 | 0.912 | 0.468 | +44.4% |
| 弱光 | 0.910 | 0.305 | +60.5% |

### 可扩展性（Exp4）

| 机器人数 | SCCS 覆盖率 | Greedy 覆盖率 | SCCS 步数 | Greedy 步数 |
|----------|-------------|---------------|-----------|-------------|
| N=3 | 0.861 | 0.764 | 273 | 335 |
| N=5 | 0.911 | 0.858 | 193 | 258 |
| N=7 | 0.923 | 0.901 | 171 | 234 |
| N=9 | 0.922 | 0.924 | 164 | 190 |

---

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd rescue_sim

# 创建虚拟环境（可选）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行实验

```bash
# 运行所有实验（30 seeds，约 40 分钟）
python run_experiments.py

# 快速验证（5 seeds，约 5 分钟）
python run_experiments.py --quick

# 运行单个实验
python run_experiments.py --exp sota       # 仅 SOTA 对比
python run_experiments.py --exp ablation   # 仅消融实验
python run_experiments.py --exp extreme    # 仅极端环境
python run_experiments.py --exp scale      # 仅可扩展性
```

### 感知模块测试

```bash
python test_perception.py
```

---

## ROS2 分布式部署

### 使用 Docker Compose

```bash
# 启动 ROS2 Humble 容器 + 5 机器人仿真
docker-compose up

# Foxglove 可视化界面
# 打开浏览器访问 http://localhost:8765
```

### 手动编译 ROS2 工作空间

```bash
cd ros2_ws
colcon build --packages-select rescue_sccs
source install/setup.bash
ros2 launch rescue_sccs rescue.launch.py
```

---

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{sccs2025,
  title   = {复杂环境救援机器人聚类搜索方法},
  author  = {},
  journal = {},
  year    = {2025},
  note    = {State-aware Clustering Search Method for Rescue Robots in Complex Environments}
}
```

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## 贡献者

本项目由以下贡献者共同完成：

| 排名 | 贡献者 | 角色 |
|------|--------|------|
| 🥇 | [Lurk0526](https://github.com/Lurk0526) | 项目主导 · 算法设计 · 实验执行 |
| 🥈 | Claude (Anthropic) | 代码实现 · 系统架构 · 论文撰写辅助 |
| 🥉 | ChatGPT (OpenAI) | 文献调研 · 实验设计建议 |
| 4 | DeepSeek | 代码审查 · 性能优化建议 |

**致谢**：本项目在研究和开发过程中得到了多个 AI 系统的辅助，特此致谢。
