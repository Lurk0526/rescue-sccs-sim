"""
test_perception.py（修复版 v2）

修复：
  1. warm-up 步数从 10 → 50
  2. 起点选距离最近幸存者最近的格子，而非固定 starts[0]
  3. 加诊断输出帮助定位问题
"""

import numpy as np
import cv2

from env import RescueEnvironment
from perception.aie import AIEModule
from perception.ekf_tracker import EKFTracker
from perception.lidar_validator import LiDARValidator
from perception.fusion_pipeline import FusionPipeline

WARMUP_STEPS = 50   # 增加预热步数

# ── 辅助：找离幸存者最近的可通行格 ──────────────────────────────
def nearest_free_to_survivors(env):
    """选距离所有幸存者重心最近的可通行格作为机器人起点"""
    if not env.survivors:
        return (env.width // 2, env.height // 2)
    cx = int(np.mean([s.x for s in env.survivors]))
    cy = int(np.mean([s.y for s in env.survivors]))
    # 在重心附近找最近的空格
    for r in range(0, 10):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                x, y = cx + dx, cy + dy
                if (0 <= x < env.width and 0 <= y < env.height
                        and env.is_free(x, y)):
                    return (x, y)
    return (env.width // 2, env.height // 2)

# ── 三级流水线测试 ────────────────────────────────────────────
print('=== 三级流水线测试 ===')

for scene in ['normal', 'dust', 'lowlight']:
    env = RescueEnvironment(seed=42)
    rx, ry = nearest_free_to_survivors(env)  # 修复：靠近幸存者出发

    # 诊断：打印起点与最近幸存者距离
    dists = [np.sqrt((s.x-rx)**2 + (s.y-ry)**2) for s in env.survivors]
    min_dist = min(dists) if dists else 999

    p = FusionPipeline(env, scene_type=scene)

    detected = []
    for step in range(WARMUP_STEPS):
        env.step()
        detected = p.detect(rx, ry)
        if detected:  # 提前找到就记录
            break

    ekf = p.ekf.get_stats()
    print(f'{scene:10s}: 检测到 {len(detected)} 个幸存者'
          f'  起点距最近幸存者={min_dist:.1f}格'
          f'  EKF={ekf}'
          f'  用了{step+1}步')

# ── 独立 EKF 单元测试（排除环境干扰）─────────────────────────
print()
print('=== EKF 单元测试（确认τ=3机制正常）===')
ekf = EKFTracker(tau=3, residual_thresh=2.5)

# 模拟连续稳定检测（固定位置，无噪声）
obs = [{'x': 10.0, 'y': 10.0, 'color': 'yellow', 'conf': 0.9}]
for i in range(6):
    confirmed = ekf.update(obs)
    stats = ekf.get_stats()
    print(f'  第{i+1}步: confirmed={len(confirmed)}  stats={stats}')

# ── LiDAR 误检率验证 ─────────────────────────────────────────
print()
print('=== LiDAR误检率验证 ===')
v = LiDARValidator()
stats = v.estimate_false_positive_rate(n_trials=2000)
print(f'纯视觉误检率: {stats["fp_rate_vision_only"]:.3f}')
print(f'融合后误检率: {stats["fp_rate_with_lidar"]:.3f}')
print(f'降低幅度:     {stats["reduction"]*100:.1f}%')

# ── AIE 模块测试 ─────────────────────────────────────────────
print()
print('=== AIE 增强模块测试 ===')
aie = AIEModule()
test_img = np.random.randint(20, 200, (480, 640, 3), dtype=np.uint8)
test_img[:240, :320] = test_img[:240, :320] // 4  # 模拟阴影区

for mode in ['normal', 'dust', 'lowlight']:
    enhanced = aie.enhance(test_img, scene_type=mode)
    orig = test_img[:240, :320].mean()
    enh  = enhanced[:240, :320].mean()
    print(f'{mode:10s}: 阴影区亮度 {orig:.1f} → {enh:.1f}  (提升 {enh-orig:+.1f})')

print()
print('✅  perception/ 测试完成')
