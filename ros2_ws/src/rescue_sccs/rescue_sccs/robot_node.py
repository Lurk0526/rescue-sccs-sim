#!/usr/bin/env python3
"""
robot_node.py
单台机器人节点：感知、导航、救援、TCFM交互
参数 robot_id 区分不同机器人实例
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import json
import numpy as np
import heapq


def astar(grid, start, goal, w, h):
    if start == goal:
        return []
    def hh(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    open_h = [(hh(start), start)]
    came, g = {}, {start: 0}
    while open_h:
        _, cur = heapq.heappop(open_h)
        if cur == goal:
            path = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            return list(reversed(path))
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nb = (cur[0]+dx, cur[1]+dy)
            if 0<=nb[0]<w and 0<=nb[1]<h:
                ng = g[cur] + 1
                if nb not in g or ng < g[nb]:
                    came[nb] = cur
                    g[nb] = ng
                    heapq.heappush(open_h, (ng+hh(nb), nb))
    return []


class RobotNode(Node):
    def __init__(self):
        super().__init__('robot_node')
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('start_x', 5.0)
        self.declare_parameter('start_y', 5.0)
        self.declare_parameter('lidar_range', 8)
        self.declare_parameter('detect_prob', 0.88)

        self.robot_id = self.get_parameter('robot_id').value
        self.pos = [self.get_parameter('start_x').value,
                    self.get_parameter('start_y').value]
        self.lidar_range = self.get_parameter('lidar_range').value
        self.detect_prob = self.get_parameter('detect_prob').value

        self.task_queue = []
        self.target = None
        self.survivors = []
        self.tcfm_done = set()
        self.rescued_count = 0
        self.visited = set()
        self._path_cache = {}

        # 发布者
        self.pos_pub     = self.create_publisher(String, '/robot/positions', 10)
        self.rescue_pub  = self.create_publisher(String, '/robot/rescue_result', 10)
        self.tcfm_pub    = self.create_publisher(String, '/tcfm/markers', 10)
        self.marker_pub  = self.create_publisher(MarkerArray, '/robot/visualization', 10)

        # 订阅者
        self.env_sub  = self.create_subscription(
            String, '/env/survivor_states', self.env_callback, 10)
        self.task_sub = self.create_subscription(
            String, '/sccs/task_assignment', self.task_callback, 10)
        self.tcfm_sub = self.create_subscription(
            String, '/tcfm/markers', self.tcfm_callback, 10)

        # 主循环 10Hz
        self.timer = self.create_timer(0.1, self.step)

        self.get_logger().info(
            f'Robot {self.robot_id} started at {self.pos}')

    def env_callback(self, msg):
        data = json.loads(msg.data)
        self.survivors = data['survivors']

    def task_callback(self, msg):
        assignment = json.loads(msg.data)
        my_tasks = assignment.get(str(self.robot_id), [])
        # 只添加未完成的任务
        self.task_queue = [t for t in my_tasks if t not in self.tcfm_done]

    def tcfm_callback(self, msg):
        data = json.loads(msg.data)
        for sid_str, state in data.items():
            sid = int(sid_str)
            if state in ('green', 'red'):
                self.tcfm_done.add(sid)
        # 清除已完成任务
        self.task_queue = [t for t in self.task_queue if t not in self.tcfm_done]
        if (self.target is not None and
                self.target.get('id') in self.tcfm_done):
            self.target = None

    def step(self):
        if not self.survivors:
            return

        # 感知（贝叶斯融合，简化版）
        detected = self._sense()

        # 处理发现的幸存者（放置黄色TCFM标记）
        for s in detected:
            if s['id'] not in self.tcfm_done:
                tcfm_msg = String()
                tcfm_msg.data = json.dumps({str(s['id']): 'yellow'})
                self.tcfm_pub.publish(tcfm_msg)

        # 选目标
        self._select_target()

        # 移动
        if self.target:
            tx, ty = int(self.target['x']), int(self.target['y'])
            px, py = int(self.pos[0]), int(self.pos[1])
            path = self._path_cache.get((px, py, tx, ty))
            if path is None:
                path = astar(None, (px, py), (tx, ty), 80, 80)
                self._path_cache[(px, py, tx, ty)] = path
            if path:
                self.pos[0] = float(path[0][0])
                self.pos[1] = float(path[0][1])
            else:
                # 直线移动
                dx = np.sign(tx - self.pos[0])
                dy = np.sign(ty - self.pos[1])
                self.pos[0] = np.clip(self.pos[0] + dx, 0, 79)
                self.pos[1] = np.clip(self.pos[1] + dy, 0, 79)

        self.visited.add((int(self.pos[0]), int(self.pos[1])))

        # 尝试救援
        self._try_rescue()

        # 发布位置
        pos_msg = String()
        pos_msg.data = json.dumps({str(self.robot_id): self.pos})
        self.pos_pub.publish(pos_msg)

        # 发布机器人可视化标记
        self._publish_robot_marker()

    def _sense(self):
        detected = []
        for s in self.survivors:
            if s['state'] != 'yellow' or s['id'] in self.tcfm_done:
                continue
            dist = np.sqrt((s['x']-self.pos[0])**2 + (s['y']-self.pos[1])**2)
            if dist > self.lidar_range:
                continue
            p_lidar = np.exp(-0.5 * (dist / self.lidar_range) ** 2)
            p_cam = self.detect_prob if dist <= 5 else self.detect_prob * 0.35
            num = p_lidar * p_cam
            den = num + (1-p_lidar)*(1-p_cam)
            if np.random.rand() < num/(den+1e-9):
                detected.append(s)
        return detected

    def _select_target(self):
        # 优先任务队列
        while self.task_queue:
            sid = self.task_queue[0]
            if sid in self.tcfm_done:
                self.task_queue.pop(0)
                continue
            s = next((x for x in self.survivors if x['id'] == sid), None)
            if s and s['state'] == 'yellow':
                self.target = s
                return
            self.task_queue.pop(0)

        # 其次感知到的最近高优先级目标
        detected = self._sense()
        valid = [s for s in detected
                 if s['state'] == 'yellow' and s['id'] not in self.tcfm_done]
        if valid:
            best = max(valid, key=lambda s:
                s['life'] / (np.sqrt((s['x']-self.pos[0])**2 +
                                     (s['y']-self.pos[1])**2) + 1))
            self.target = best
            return

        # 探索
        if self.target is None:
            for _ in range(30):
                x = np.random.randint(1, 79)
                y = np.random.randint(1, 79)
                if (x, y) not in self.visited:
                    self.target = {'id': -1, 'x': float(x), 'y': float(y)}
                    break

    def _try_rescue(self):
        if self.target is None or self.target.get('id', -1) < 0:
            return
        sid = self.target['id']
        s = next((x for x in self.survivors if x['id'] == sid), None)
        if not s or s['state'] != 'yellow':
            return
        dist = np.sqrt((s['x']-self.pos[0])**2 + (s['y']-self.pos[1])**2)
        if dist <= 1.5:
            result = 'green' if s['life'] > 0.05 else 'red'
            # 发布救援结果给env_node
            rescue_msg = String()
            rescue_msg.data = json.dumps({
                'survivor_id': sid,
                'result': result,
                'robot_id': self.robot_id,
            })
            self.rescue_pub.publish(rescue_msg)
            # 发布TCFM标记
            tcfm_msg = String()
            tcfm_msg.data = json.dumps({str(sid): result})
            self.tcfm_pub.publish(tcfm_msg)
            self.rescued_count += 1
            self.target = None
            self.task_queue = [t for t in self.task_queue if t != sid]
            self.get_logger().info(
                f'Robot {self.robot_id} rescued survivor {sid} → {result}'
                f' (total: {self.rescued_count})')

    def _publish_robot_marker(self):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f'robot_{self.robot_id}'
        m.id = self.robot_id
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.position.x = self.pos[0] * 0.5
        m.pose.position.y = self.pos[1] * 0.5
        m.pose.position.z = 0.3
        m.scale.x = 0.8
        m.scale.y = 0.2
        m.scale.z = 0.2
        colors = [(0.1,0.4,0.8),(0.8,0.2,0.2),(0.2,0.7,0.2),
                  (0.8,0.6,0.1),(0.6,0.2,0.8)]
        r, g, b = colors[self.robot_id % len(colors)]
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
        arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = RobotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
