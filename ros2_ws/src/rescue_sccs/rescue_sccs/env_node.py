#!/usr/bin/env python3
"""
env_node.py
环境管理节点：幸存者状态、生命强度衰减、地图信息发布
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json
import numpy as np
import sys
import os

sys.path.insert(0, '/root/output')


class EnvNode(Node):
    def __init__(self):
        super().__init__('env_node')
        self.get_logger().info('EnvNode starting...')

        # 参数
        self.declare_parameter('width', 80)
        self.declare_parameter('height', 80)
        self.declare_parameter('n_survivors', 50)
        self.declare_parameter('n_clusters', 5)
        self.declare_parameter('life_decay_rate', 0.0018)
        self.declare_parameter('seed', 42)

        w = self.get_parameter('width').value
        h = self.get_parameter('height').value
        n = self.get_parameter('n_survivors').value
        k = self.get_parameter('n_clusters').value
        decay = self.get_parameter('life_decay_rate').value
        seed = self.get_parameter('seed').value

        # 初始化环境（复用Python仿真代码）
        np.random.seed(seed)
        self.survivors = self._gen_survivors(w, h, n, k)
        self.width = w
        self.height = h
        self.decay = decay
        self.step = 0

        # 发布者
        self.state_pub = self.create_publisher(String, '/env/survivor_states', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/env/visualization', 10)

        # 订阅者（接收机器人的救援结果）
        self.rescue_sub = self.create_subscription(
            String, '/robot/rescue_result', self.rescue_callback, 10)

        # 定时器：每0.1秒推进一步
        self.timer = self.create_timer(0.1, self.step_callback)
        self.get_logger().info(f'Environment initialized: {n} survivors in {w}x{h} grid')

    def _gen_survivors(self, w, h, n, k):
        centers = [(np.random.randint(10, w-10), np.random.randint(10, h-10))
                   for _ in range(k)]
        survivors = []
        per = n // k
        sid = 0
        for cx, cy in centers:
            count = per if sid < n - per else n - sid
            for _ in range(count):
                x = int(np.clip(cx + np.random.randn()*5, 1, w-2))
                y = int(np.clip(cy + np.random.randn()*5, 1, h-2))
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                intensity = float(np.clip(
                    max(0.25, 1.0 - dist/12) + np.random.randn()*0.1, 0.1, 1.0))
                survivors.append({
                    'id': sid, 'x': float(x), 'y': float(y),
                    'life': intensity, 'state': 'yellow'
                })
                sid += 1
                if sid >= n:
                    break
            if sid >= n:
                break
        return survivors

    def step_callback(self):
        self.step += 1
        # 生命衰减
        for s in self.survivors:
            if s['state'] == 'yellow':
                s['life'] = max(0.0, s['life'] - self.decay)
                if s['life'] <= 0:
                    s['state'] = 'red'

        # 发布状态
        msg = String()
        msg.data = json.dumps({
            'step': self.step,
            'survivors': self.survivors
        })
        self.state_pub.publish(msg)

        # 发布可视化标记
        self.publish_markers()

        if self.step % 50 == 0:
            rescued = sum(1 for s in self.survivors if s['state'] == 'green')
            dead = sum(1 for s in self.survivors if s['state'] == 'red')
            self.get_logger().info(
                f'Step {self.step}: rescued={rescued} dead={dead} '
                f'coverage={rescued/len(self.survivors):.2%}')

    def rescue_callback(self, msg):
        data = json.loads(msg.data)
        sid = data['survivor_id']
        result = data['result']
        if 0 <= sid < len(self.survivors):
            if self.survivors[sid]['state'] == 'yellow':
                self.survivors[sid]['state'] = result
                self.get_logger().info(
                    f'Survivor {sid} marked as {result}')

    def publish_markers(self):
        arr = MarkerArray()
        color_map = {
            'yellow': (1.0, 1.0, 0.0),
            'green':  (0.0, 1.0, 0.0),
            'red':    (1.0, 0.0, 0.0),
        }
        for s in self.survivors:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'survivors'
            m.id = s['id']
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = s['x'] * 0.5
            m.pose.position.y = s['y'] * 0.5
            m.pose.position.z = 0.3
            m.scale.x = 0.4
            m.scale.y = 0.4
            m.scale.z = 0.6
            r, g, b = color_map.get(s['state'], (0.5, 0.5, 0.5))
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.9
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = EnvNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
