"""
algorithms/sccs.py
论文提出方法：SCCS
核心：ADC自适应密度聚类 + TCFM三色荧光标记 + 雷视融合感知
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import RescueEnvironment
from tcfm import TCFMSystem
from robot import Robot, SensorConfig


# ─────────────────────────────────────────────────
# ADC 自适应密度聚类（论文3.4节）
# ─────────────────────────────────────────────────

def _adaptive_eps(n_robots: int, n_survivors: int, map_area: int) -> float:
    """
    自适应计算DBSCAN的ε参数

    推导逻辑：
      密度越高 → ε越小（幸存者集中，精细划分）
      机器人越多 → ε越大（每台负责更大区域）

    公式来自论文3.4.1节：
      ε = 8.0 × (1 + 0.4×ln(K)) / (1 + 8×density)
      其中 density = n_survivors / map_area
    """
    if n_survivors == 0:
        return 8.0
    density = n_survivors / map_area
    eps = 8.0 * (1.0 + 0.4 * np.log(max(n_robots, 1))) / (1.0 + 8.0 * density)
    return float(np.clip(eps, 4.0, 20.0))


def _do_adc_cluster(env, robots):
    """
    ADC自适应密度聚类任务分配（论文3.4节完整实现）

    步骤：
    1. 收集当前所有黄色幸存者坐标
    2. 自适应ε的DBSCAN聚类
    3. 簇内按生命强度升序排列（最危急的优先救援）
    4. 带负载均衡的最近机器人分配
       评分 = 距质心距离 + 当前负载×惩罚系数
    5. 离群点（label=-1）归入最近簇
    """
    yellow = [s for s in env.survivors if s.state == 'yellow']
    if not yellow:
        for r in robots:
            r.task_queue = []
        return

    n_robots = len(robots)
    coords = np.array([[s.x, s.y] for s in yellow], dtype=float)

    # 自适应参数
    eps = _adaptive_eps(n_robots, len(yellow), env.width * env.height)
    min_samples = max(1, len(yellow) // (n_robots * 2))

    # DBSCAN聚类
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_

    # 按label分组（-1为离群点，单独处理）
    clusters = {}
    noise_sids = []
    for i, lbl in enumerate(labels):
        if lbl == -1:
            noise_sids.append(yellow[i].id)
        else:
            clusters.setdefault(lbl, []).append(yellow[i].id)

    # 簇内按生命强度升序（最危急优先）
    for lbl in clusters:
        clusters[lbl].sort(key=lambda sid: env.survivors[sid].life_intensity)

    # 负载均衡分配
    robot_pos = np.array([[r.pos[0], r.pos[1]] for r in robots], dtype=float)
    loads = np.zeros(n_robots)

    # 清空任务队列
    for r in robots:
        r.task_queue = []

    # 大簇优先分配
    cluster_list = sorted(clusters.values(), key=len, reverse=True)
    for sids in cluster_list:
        if not sids:
            continue
        cx = np.mean([env.survivors[s].x for s in sids])
        cy = np.mean([env.survivors[s].y for s in sids])
        # 综合评分：距离 + 负载惩罚（论文中α=3.0）
        dists = cdist([[cx, cy]], robot_pos)[0]
        scores = dists + loads * 3.0
        best = int(np.argmin(scores))
        robots[best].task_queue.extend(sids)
        loads[best] += len(sids)

    # 离群点：分配给最近的负载最轻机器人（顺路救援）
    for sid in noise_sids:
        s = env.survivors[sid]
        dists = cdist([[s.x, s.y]], robot_pos)[0]
        scores = dists + loads * 3.0
        best = int(np.argmin(scores))
        robots[best].task_queue.append(sid)
        loads[best] += 1
        
    # ── 计数再均衡（论文3.4.2节动态负载均衡）
    # 确保任何两台机器人的任务数差不超过2
    # 从任务最多的机器人取出任务给任务最少的
    for _ in range(50):
        queue_lens = [len(r.task_queue) for r in robots]
        max_load = max(queue_lens)
        min_load = min(queue_lens)
        if max_load - min_load <= 2:
            break
        over_idx  = int(np.argmax(queue_lens))
        under_idx = int(np.argmin(queue_lens))
        # 取出负载最重机器人的最后一个任务（优先级最低的）
        task = robots[over_idx].task_queue.pop()
        robots[under_idx].task_queue.append(task)

# ─────────────────────────────────────────────────
# SCCS 系统
# ─────────────────────────────────────────────────

class SCCSSystem:
    """
    完整SCCS系统：
    - TCFM：三色荧光标记，O(1)通信
    - ADC：自适应密度聚类，带生命强度权重
    - 雷视融合：贝叶斯两阶段感知（由SensorConfig控制）
    - 动态重聚类：每recluster_interval步响应环境变化
    """

    def __init__(self, env, n_robots=5, sensor=None,
                 recluster_interval=40):
        self.env = env
        self.n_robots = n_robots
        self.recluster_interval = recluster_interval

        # TCFM系统（单例，所有机器人共享环境标记）
        self.tcfm = TCFMSystem()
        sensor = sensor or SensorConfig()

        # 初始化机器人（use_tcfm=True，使用TCFM通信）
        starts = env._robot_starts(n=n_robots)
        self.robots = [
            Robot(i, starts[i], env, self.tcfm, None,
                  sensor, use_tcfm=True)
            for i in range(n_robots)
        ]

        # 初始ADC聚类分配
        _do_adc_cluster(env, self.robots)

        # 指标记录
        self.coverage_curve = []
        self.response_step = -1

    def run(self, max_steps=3000):
        for step in range(max_steps):
            # 环境推进（生命衰减）
            self.env.step()

            # 定期重新聚类（动态响应幸存者状态变化）
            if step > 0 and step % self.recluster_interval == 0:
                _do_adc_cluster(self.env, self.robots)
                # 额外负载均衡：任务过多的机器人分一部分给空闲机器人
                loads = [len(r.task_queue) for r in self.robots]
                for _ in range(10):
                    max_load = max(loads)
                    min_load = min(loads)
                    if max_load - min_load <= 3:
                        break
                    over  = loads.index(max_load)
                    under = loads.index(min_load)
                    if self.robots[over].task_queue:
                        task = self.robots[over].task_queue.pop()
                        self.robots[under].task_queue.append(task)
                        loads[over]  -= 1
                        loads[under] += 1

            # 所有机器人执行一步
            for robot in self.robots:
                robot.step(self.robots)

            # 记录指标
            self.coverage_curve.append(self.env.coverage_rate)

            if self.response_step == -1:
                if any(s.state == 'green' for s in self.env.survivors):
                    self.response_step = self.env.current_step

            if self.env.is_complete:
                break

        return self._metrics()

    def _metrics(self):
        stats = self.env.get_stats()
        loads = [r.rescued_count for r in self.robots]
        return {
            'method':          'SCCS',
            'completion_step': self.env.current_step,
            'coverage_rate':   stats['coverage_rate'],
            'rescued':         stats['rescued'],
            'dead':            stats['dead'],
            'response_step':   max(self.response_step, 1),
            'load_variance':   float(np.var(loads)),
            'collision_count': sum(r.collisions for r in self.robots),
            'comm_cost':       0,                                    # TCFM无显式通信
            'marker_reads':    self.tcfm.get_stats()['reads'],       # 参考值
            'coverage_curve':  self.coverage_curve,
        }
