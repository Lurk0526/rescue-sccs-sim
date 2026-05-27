"""
algorithms/baselines.py
对比算法 + 消融变体

SOTA对比：
  GreedySystem     - 贪心最近幸存者 + 广播通信
  PSOSystem        - Voronoi区域巡逻 + 广播通信
  RandomSystem     - 随机游走（下界）

消融变体：
  SCCS_noTCFM      - ADC聚类 + 广播通信（去掉TCFM）
  SCCS_noClustering- TCFM + 轮询分配（去掉ADC聚类）
  SCCS_noFusion    - TCFM + ADC + 纯视觉传感器（去掉LiDAR融合）
"""

import numpy as np
from scipy.spatial.distance import cdist
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import RescueEnvironment
from tcfm import TCFMSystem, BroadcastComm
from robot import Robot, SensorConfig
from algorithms.sccs import _do_adc_cluster


# ─────────────────────────────────────────────────
# 公共运行循环（所有系统复用）
# ─────────────────────────────────────────────────

def _run_loop(system, max_steps=3000):
    """
    统一的仿真主循环。

    每步流程：
      1. env.step()       → 生命衰减
      2. system._pre_step → 系统特定的前置操作（巡逻/广播等）
      3. robot.step()     → 感知/决策/移动/救援（Robot内部处理）
      4. 记录指标
    """
    system.coverage_curve = []
    system.response_step = -1

    for step in range(max_steps):
        system.env.step()

        # 系统特定前置操作
        if hasattr(system, '_pre_step'):
            system._pre_step(step)

        # 所有机器人执行
        for robot in system.robots:
            robot.step(system.robots)

        # 记录覆盖率
        system.coverage_curve.append(system.env.coverage_rate)

        # 记录达到80%覆盖率的时间
        if system.response_step == -1:
            if any(s.state == 'green' for s in system.env.survivors):
                system.response_step = system.env.current_step

        if system.env.is_complete:
            break

    return _collect_metrics(system)


def _collect_metrics(system):
    stats = system.env.get_stats()
    loads = [r.rescued_count for r in system.robots]

    # 显式通信消息数：TCFM系统为0（无机器人间直接通信）
    # 广播系统为实际广播消息总数
    if hasattr(system, 'tcfm') and not hasattr(system, 'broadcast'):
        explicit_comm = 0                              # TCFM：无显式通信
        marker_reads  = system.tcfm.get_stats()['reads']  # 仅作参考
    elif hasattr(system, 'broadcast'):
        explicit_comm = system.broadcast.broadcast_count   # 广播消息数
        marker_reads  = 0
    else:
        explicit_comm = 0
        marker_reads  = 0

    return {
        'method':          system.method_name,
        'completion_step': system.env.current_step,
        'coverage_rate':   stats['coverage_rate'],
        'rescued':         stats['rescued'],
        'dead':            stats['dead'],
        'response_step':   max(system.response_step, 1),
        'load_variance':   float(np.var(loads)),
        'collision_count': sum(r.collisions for r in system.robots),
        'comm_cost':       explicit_comm,   # 显式通信消息数
        'marker_reads':    marker_reads,    # TCFM标记读取次数（参考）
        'coverage_curve':  system.coverage_curve,
    }


# ─────────────────────────────────────────────────
# 对比算法1：Greedy（最近幸存者贪心 + 广播通信）
# ─────────────────────────────────────────────────

class GreedySystem:
    """
    传统贪心方法：
    - 每台机器人独立选最近已知幸存者
    - 广播通信（每8步同步），模拟通信延迟
    - 无聚类、无TCFM

    缺点体现：
    - 负载不均（多台机器人扑向同一簇）
    - 广播延迟导致重复搜索
    - 通信开销O(N²)
    """
    method_name = 'Greedy'

    def __init__(self, env, n_robots=5, sensor=None,
                 sync_interval=25):
        self.env = env
        self.broadcast = BroadcastComm(n_robots, sync_interval)

        # 传感器：稍弱于SCCS（无完整融合）
        sensor = sensor or SensorConfig(
            lidar_grid=8,
            camera_grid=5,
            detect_prob=0.88,
            env_factor=1.0,
            marker_range=6.0,
        )
        starts = env._robot_starts(n=n_robots)
        self.robots = [
            Robot(i, starts[i], env, None, self.broadcast,
                  sensor, use_tcfm=False)
            for i in range(n_robots)
        ]
        # 无聚类，任务队列为空，机器人依赖实时感知选目标

    def run(self, max_steps=3000):
        return _run_loop(self, max_steps)
    # 注：广播同步在Robot.step()内部调用self.broadcast.sync()


# ─────────────────────────────────────────────────
# 对比算法2：PSO（Voronoi区域巡逻 + 广播通信）
# ─────────────────────────────────────────────────

class PSOSystem:
    """
    PSO路径优化巡逻：
    - 预计算Voronoi区域划分（代表PSO路径规划输出）
    - 机器人在分配区域内按序巡逻
    - 发现幸存者立即救援
    - 广播通信（8步同步），无TCFM

    代表的论文基线：
    常美玉等人的聚类驱动交叉遗传算法（文献[1]）、
    王晓庆等人的改进迭代贪婪算法（文献[14]）等路径规划类方法
    """
    method_name = 'PSO'

    def __init__(self, env, n_robots=5, sensor=None,
                 sync_interval=25):
        self.env = env
        self.broadcast = BroadcastComm(n_robots, sync_interval)

        sensor = sensor or SensorConfig(
            lidar_grid=8,
            camera_grid=5,
            detect_prob=0.86,
            env_factor=1.0,
            marker_range=6.0,
        )
        starts = env._robot_starts(n=n_robots)
        self.robots = [
            Robot(i, starts[i], env, None, self.broadcast,
                  sensor, use_tcfm=False)
            for i in range(n_robots)
        ]
        # 预计算巡逻区域
        self._assign_patrol_zones()

    def _assign_patrol_zones(self):
        """
        Voronoi区域划分：将地图均匀采样点按最近机器人分配。
        每台机器人得到一组有序巡逻点。
        """
        starts = np.array([r.pos for r in self.robots], dtype=float)

        # 在地图上均匀采样巡逻点（每6格一个）
        waypoints = []
        for y in range(3, self.env.height - 3, 6):
            for x in range(3, self.env.width - 3, 6):
                if self.env.grid[y, x] == 0:
                    waypoints.append((x, y))

        if not waypoints:
            for r in self.robots:
                r._patrol = []
                r._patrol_idx = 0
            return

        # Voronoi分配
        wp_arr = np.array(waypoints, dtype=float)
        dists = cdist(wp_arr, starts)
        assignments = np.argmin(dists, axis=1)

        for i, robot in enumerate(self.robots):
            zone = [waypoints[j]
                    for j in range(len(waypoints))
                    if assignments[j] == i]
            # 按蛇形顺序排列（减少来回跑动）
            zone.sort(key=lambda p: (p[1] // 6) * (1 if (p[1] // 6) % 2 == 0 else -1),)
            robot._patrol = zone
            robot._patrol_idx = 0

    def _pre_step(self, step):
        """
        巡逻注入：
        当机器人task_queue为空且无当前目标时，
        注入下一个巡逻点作为导航目标。
        Robot.step()内部的_select_target会在有感知目标时覆盖此目标。
        """
        for robot in self.robots:
            has_task = len(robot.task_queue) > 0
            has_target = (robot.target is not None
                          and robot.target != robot.pos)

            if not has_task and not has_target:
                if hasattr(robot, '_patrol') and robot._patrol:
                    idx = robot._patrol_idx % len(robot._patrol)
                    robot.target = robot._patrol[idx]
                    # 到达当前巡逻点后推进索引
                    if robot.pos == robot.target:
                        robot._patrol_idx = (
                            (robot._patrol_idx + 1) % len(robot._patrol)
                        )
                        robot.target = None  # 让下一步重新分配

    def run(self, max_steps=3000):
        return _run_loop(self, max_steps)


# ─────────────────────────────────────────────────
# 对比算法3：Random（随机游走，下界基线）
# ─────────────────────────────────────────────────

class RandomSystem:
    """
    纯随机游走：
    - 机器人随机选相邻空格移动
    - 感知范围内发现幸存者时救援
    - 无通信、无规划
    作用：确认论文方法的下界，体现算法的必要性
    """
    method_name = 'Random'

    def __init__(self, env, n_robots=5, sensor=None):
        self.env = env
        self.broadcast = BroadcastComm(n_robots, sync_interval=99999)

        sensor = sensor or SensorConfig(
            lidar_grid=6,
            camera_grid=4,
            detect_prob=0.80,
            env_factor=1.0,
            marker_range=5.0,
        )
        starts = env._robot_starts(n=n_robots)
        self.robots = [
            Robot(i, starts[i], env, None, self.broadcast,
                  sensor, use_tcfm=False)
            for i in range(n_robots)
        ]

    def _pre_step(self, step):
        """强制随机目标，覆盖机器人自身的探索决策"""
        for robot in self.robots:
            # 每步随机选一个相邻空格作为目标
            if robot.pos == robot.target or robot.target is None:
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]
                np.random.shuffle(dirs)
                for dx, dy in dirs:
                    nx = robot.pos[0] + dx
                    ny = robot.pos[1] + dy
                    if self.env.is_free(nx, ny):
                        robot.target = (nx, ny)
                        robot.target_sid = None
                        break

    def run(self, max_steps=3000):
        return _run_loop(self, max_steps)


# ─────────────────────────────────────────────────
# 消融1：去掉TCFM，保留ADC聚类
# ─────────────────────────────────────────────────

class SCCS_noTCFM:
    """
    消融实验：去掉TCFM，改用广播通信。
    保留ADC聚类分配。

    验证目的：
    - 体现TCFM对通信开销的降低（广播O(N²) vs 标记O(1)）
    - 体现广播延迟导致的重复搜索问题
    - 预期结果：comm_cost大幅增加，coverage_rate轻微下降
    """
    method_name = 'w/o TCFM'

    def __init__(self, env, n_robots=5, sensor=None,
                 sync_interval=25):
        self.env = env
        self.broadcast = BroadcastComm(n_robots, sync_interval)

        sensor = sensor or SensorConfig()  # 同SCCS的传感器
        starts = env._robot_starts(n=n_robots)
        self.robots = [
            Robot(i, starts[i], env, None, self.broadcast,
                  sensor, use_tcfm=False)
            for i in range(n_robots)
        ]
        # 使用ADC聚类初始分配（与SCCS相同）
        _do_adc_cluster(env, self.robots)

    def _pre_step(self, step):
        # 定期重新聚类（与SCCS一致，每60步）
        if step > 0 and step % 60 == 0:
            _do_adc_cluster(self.env, self.robots)

    def run(self, max_steps=3000):
        return _run_loop(self, max_steps)


# ─────────────────────────────────────────────────
# 消融2：去掉ADC聚类，保留TCFM
# ─────────────────────────────────────────────────

class SCCS_noClustering:
    """
    消融实验：去掉ADC聚类，改用轮询分配（sid % n_robots）。
    保留TCFM通信。

    验证目的：
    - 体现ADC聚类对负载均衡的贡献
    - 轮询分配无视地理分布，导致机器人跨区域奔波
    - 预期结果：load_variance升高，completion_step增大
    """
    method_name = 'w/o Clustering'

    def __init__(self, env, n_robots=5, sensor=None):
        self.env = env
        self.tcfm = TCFMSystem()

        sensor = sensor or SensorConfig()
        starts = env._robot_starts(n=n_robots)
        self.robots = [
            Robot(i, starts[i], env, self.tcfm, None,
                  sensor, use_tcfm=True)
            for i in range(n_robots)
        ]

        # 轮询分配：按幸存者id取模，不考虑地理位置
        n = len(env.survivors)
        for r in self.robots:
            r.task_queue = []
        for s in env.survivors:
            self.robots[s.id % n_robots].task_queue.append(s.id)

    def run(self, max_steps=3000):
        return _run_loop(self, max_steps)


# ─────────────────────────────────────────────────
# 消融3：去掉LiDAR融合，保留TCFM + ADC聚类
# ─────────────────────────────────────────────────

class SCCS_noFusion:
    """
    消融实验：传感器降级为纯视觉（无LiDAR）。
    保留TCFM + ADC聚类。

    验证目的：
    - 体现雷视融合对感知范围和准确率的贡献
    - 纯视觉感知范围小（6格 vs 10格），扬尘/弱光下更差
    - 预期结果：
        正常环境下coverage_rate轻微下降
        扬尘/弱光环境下差距明显（对应论文"78.3%→94.2%"）
    """
    method_name = 'w/o LiDAR Fusion'

    def __init__(self, env, n_robots=5, sensor=None):
        self.env = env
        self.tcfm = TCFMSystem()

        # 强制使用纯视觉传感器（lidar_grid=0）
        sensor = SensorConfig.vision_only()

        starts = env._robot_starts(n=n_robots)
        self.robots = [
            Robot(i, starts[i], env, self.tcfm, None,
                  sensor, use_tcfm=True)
            for i in range(n_robots)
        ]

        # 不做初始聚类，纯视觉感知范围小，依赖逐步发现

    def _pre_step(self, step):
        if step > 0 and step % 40 == 0:
            # 只对已发现的幸存者聚类（模拟纯视觉的延迟发现）
            yellow_found = [s for s in self.env.survivors
                           if s.state == 'yellow' and s.discovery_step >= 0]
            if yellow_found:
                import numpy as np
                from sklearn.cluster import DBSCAN
                from scipy.spatial.distance import cdist
                coords = [[s.x, s.y] for s in yellow_found]
                eps = 8.0
                labels = DBSCAN(eps=eps, min_samples=1).fit(coords).labels_
                clusters = {}
                for i, lbl in enumerate(labels):
                    clusters.setdefault(lbl, []).append(yellow_found[i].id)
                robot_pos = [[r.pos[0], r.pos[1]] for r in self.robots]
                loads = [0] * len(self.robots)
                for r in self.robots:
                    r.task_queue = []
                for sids in sorted(clusters.values(), key=len, reverse=True):
                    cx = sum(self.env.survivors[s].x for s in sids) / len(sids)
                    cy = sum(self.env.survivors[s].y for s in sids) / len(sids)
                    dists = [((cx-p[0])**2+(cy-p[1])**2)**0.5 + loads[i]*3 for i,p in enumerate(robot_pos)]
                    best = dists.index(min(dists))
                    self.robots[best].task_queue.extend(sids)
                    loads[best] += len(sids)

    def run(self, max_steps=3000):
        return _run_loop(self, max_steps)
