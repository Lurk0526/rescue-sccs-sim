#!/usr/bin/env python3
"""
sccs_node.py
SCCS协调节点：ADC聚类任务分配 + TCFM状态驱动
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import numpy as np

try:
    from sklearn.cluster import DBSCAN
    from scipy.spatial.distance import cdist
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SCCSNode(Node):
    def __init__(self):
        super().__init__('sccs_node')
        self.declare_parameter('n_robots', 5)
        self.n_robots = self.get_parameter('n_robots').value

        # 机器人当前位置（从各robot_node接收）
        self.robot_positions = {i: [float(5*(i+1)), 5.0] for i in range(self.n_robots)}
        self.robot_tasks = {i: [] for i in range(self.n_robots)}

        # TCFM标记状态
        self.tcfm_markers = {}  # sid -> 'yellow'/'green'/'red'

        # 发布者：向各机器人发布任务
        self.task_pub = self.create_publisher(String, '/sccs/task_assignment', 10)

        # 订阅者
        self.env_sub = self.create_subscription(
            String, '/env/survivor_states', self.env_callback, 10)
        self.pos_sub = self.create_subscription(
            String, '/robot/positions', self.pos_callback, 10)
        self.tcfm_sub = self.create_subscription(
            String, '/tcfm/markers', self.tcfm_callback, 10)

        # 定期重聚类（40步 = 4秒）
        self.step_count = 0
        self.timer = self.create_timer(4.0, self.recluster)

        self.get_logger().info(f'SCCS Node started with {self.n_robots} robots')

    def env_callback(self, msg):
        data = json.loads(msg.data)
        self.step_count = data['step']
        self.survivors = data['survivors']

    def pos_callback(self, msg):
        data = json.loads(msg.data)
        for robot_id, pos in data.items():
            self.robot_positions[int(robot_id)] = pos

    def tcfm_callback(self, msg):
        data = json.loads(msg.data)
        self.tcfm_markers.update(data)

    def recluster(self):
        if not hasattr(self, 'survivors'):
            return

        # 只对待救（yellow）且未被TCFM标记完成的幸存者做聚类
        yellow = [s for s in self.survivors
                  if s['state'] == 'yellow'
                  and self.tcfm_markers.get(s['id'], 'yellow') == 'yellow']

        if not yellow or not HAS_SKLEARN:
            return

        coords = np.array([[s['x'], s['y']] for s in yellow])
        n = len(yellow)
        density = n / (80 * 80)
        eps = float(np.clip(
            8.0 * (1 + 0.4 * np.log(self.n_robots)) / (1 + 8 * density),
            4.0, 20.0))

        labels = DBSCAN(eps=eps, min_samples=max(1, n//(self.n_robots*2))).fit(coords).labels_

        clusters = {}
        for i, lbl in enumerate(labels):
            clusters.setdefault(lbl, []).append(yellow[i]['id'])

        robot_pos = np.array([self.robot_positions.get(i, [5.0, 5.0])
                              for i in range(self.n_robots)])
        loads = np.zeros(self.n_robots)
        assignment = {i: [] for i in range(self.n_robots)}

        for sids in sorted(clusters.values(), key=len, reverse=True):
            s_coords = [[self.survivors[sid]['x'], self.survivors[sid]['y']]
                        for sid in sids if sid < len(self.survivors)]
            if not s_coords:
                continue
            cx = np.mean([c[0] for c in s_coords])
            cy = np.mean([c[1] for c in s_coords])
            dists = cdist([[cx, cy]], robot_pos)[0] + loads * 3.0
            best = int(np.argmin(dists))
            assignment[best].extend(sids)
            loads[best] += len(sids)

        # 发布任务分配
        msg = String()
        msg.data = json.dumps(assignment)
        self.task_pub.publish(msg)
        self.get_logger().info(
            f'Reclustered: {len(yellow)} targets → '
            f'{[len(v) for v in assignment.values()]} per robot')


def main(args=None):
    rclpy.init(args=args)
    node = SCCSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
