import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathfinding import next_step


def fused_detect(env, pos, sensor):
    x, y = pos
    confirmed = []
    for s in env.survivors:
        if s.state != 'yellow':
            continue
        true_dist = np.sqrt((s.x - x)**2 + (s.y - y)**2)
        if sensor.lidar_grid == 0:
            if true_dist <= sensor.camera_grid:
                if np.random.rand() < sensor.detect_prob * sensor.env_factor:
                    confirmed.append(s)
            continue
        lidar_dist = true_dist + np.random.randn() * sensor.lidar_noise_std
        if lidar_dist > sensor.lidar_grid:
            continue
        p_lidar = float(np.exp(-0.5 * (true_dist / sensor.lidar_grid) ** 2))
        if true_dist <= sensor.camera_grid:
            p_camera = sensor.detect_prob * sensor.env_factor
        else:
            p_camera = sensor.detect_prob * sensor.env_factor * 0.35
        numerator   = p_lidar * p_camera
        denominator = numerator + (1 - p_lidar) * (1 - p_camera)
        p_fused = numerator / (denominator + 1e-9)
        if np.random.rand() < p_fused:
            confirmed.append(s)
    return confirmed


@dataclass
class SensorConfig:
    lidar_real_range:  float = 12.0
    lidar_decay:       float = 0.42
    lidar_grid:        int   = 8
    lidar_noise_std:   float = 1.0
    camera_real_range: float = 10.0
    camera_decay:      float = 0.30
    camera_grid:       int   = 5
    detect_prob:       float = 0.88
    false_positive:    float = 0.04
    env_factor:        float = 1.0
    effective_range:   float = 8.0
    marker_range:      float = 6.0
    rescue_radius:     float = 1.5

    @classmethod
    def vision_only(cls):
        return cls(
            lidar_grid=0,
            lidar_noise_std=0.0,
            camera_grid=5,
            detect_prob=0.75,
            false_positive=0.06,
            env_factor=1.0,
            effective_range=5.0,
            marker_range=5.0,
            rescue_radius=1.5,
        )

    @classmethod
    def dusty(cls):
        return cls(
            lidar_grid=7,
            lidar_noise_std=1.2,
            camera_grid=3,
            detect_prob=0.75,
            false_positive=0.06,
            env_factor=0.55,
            effective_range=7.0,
            marker_range=5.0,
            rescue_radius=1.5,
        )

    @classmethod
    def low_light(cls):
        return cls(
            lidar_grid=8,
            lidar_noise_std=1.0,
            camera_grid=2,
            detect_prob=0.75,
            false_positive=0.10,
            env_factor=0.40,
            effective_range=8.0,
            marker_range=6.0,
            rescue_radius=1.5,
        )


class Robot:
    def __init__(self, robot_id, start, env, tcfm, broadcast,
                 sensor=None, use_tcfm=True):
        self.id = robot_id
        self.pos = start
        self.env = env
        self.tcfm = tcfm
        self.broadcast = broadcast
        self.sensor = sensor or SensorConfig()
        self.use_tcfm = use_tcfm
        self.target = None
        self.target_sid = None
        self.task_queue = []
        self.rescued_count = 0
        self.steps_moved = 0
        self.collisions = 0
        self.path_history = [start]
        self.visited = np.zeros((env.height, env.width), dtype=bool)
        self._cache = {}

    def step(self, all_robots):
        detected = fused_detect(self.env, self.pos, self.sensor)

        r = int(max(self.sensor.lidar_grid, self.sensor.camera_grid))
        x0, y0 = self.pos
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x0 + dx, y0 + dy
                if (0 <= nx < self.env.width and
                        0 <= ny < self.env.height and
                        np.sqrt(dx**2 + dy**2) <= r):
                    self.visited[ny, nx] = True

        for s in detected:
            self.env.mark_discovered(s.id)
            if self.use_tcfm and self.tcfm and not self.tcfm.is_marked(s.id):
                self.tcfm.place_yellow(
                    s.id, s.x, s.y, self.id, self.env.current_step)
            elif self.broadcast:
                self.broadcast.local_update(self.id, s.id, 'yellow')

        if self.use_tcfm and self.tcfm:
            markers = self.tcfm.read_in_range(
                *self.pos, self.sensor.marker_range, self.id)
            done = {m.survivor_id for m in markers
                    if m.color in ('green', 'red')}
            self.task_queue = [s for s in self.task_queue if s not in done]
            if self.target_sid in done:
                self.target = None
                self.target_sid = None

        if self.broadcast:
            self.broadcast.sync(self.env.current_step)

        self._select_target(detected)

        if self.target and self.target != self.pos:
            nxt = next_step(self.env.grid, self.pos, self.target, self._cache)
            occupied = {r.pos for r in all_robots if r.id != self.id}
            if nxt in occupied:
                alternative = None
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                np.random.shuffle(dirs)
                for dx, dy in dirs:
                    nx, ny = self.pos[0] + dx, self.pos[1] + dy
                    if self.env.is_free(nx, ny) and (nx, ny) not in occupied:
                        alternative = (nx, ny)
                        break
                if alternative is None:
                    self.collisions += 1
                    nxt = self.pos
                else:
                    nxt = alternative
            self.pos = nxt
            self.steps_moved += 1
            self.path_history.append(self.pos)
            self.env.update_visit(*self.pos)

        self._try_rescue()

    def _select_target(self, detected):
        while self.task_queue:
            sid = self.task_queue[0]
            s = self.env.survivors[sid]
            if s.state == 'yellow':
                self.target = (s.x, s.y)
                self.target_sid = sid
                return
            self.task_queue.pop(0)

        valid = [s for s in detected if s.state == 'yellow']
        if valid:
            best = max(valid, key=lambda s:
                s.life_intensity / (
                    np.sqrt((s.x - self.pos[0])**2 +
                            (s.y - self.pos[1])**2) + 1))
            self.target = (best.x, best.y)
            self.target_sid = best.id
            return

        if self.target is None or self.pos == self.target:
            best_p, best_s = None, -1
            for _ in range(50):
                x = np.random.randint(1, self.env.width - 1)
                y = np.random.randint(1, self.env.height - 1)
                if self.env.grid[y, x] == 0:
                    unvisited_score = 0 if self.visited[y, x] else 15
                    random_score = np.random.rand() * 3
                    avoid_score = 0
                    if self.use_tcfm and self.tcfm:
                        for m in self.tcfm._markers.values():
                            if m.color in ('green', 'red'):
                                d = np.sqrt((x - m.x)**2 + (y - m.y)**2)
                                if d < 5:
                                    avoid_score -= 5
                    score = unvisited_score + random_score + avoid_score
                    if score > best_s:
                        best_s = score
                        best_p = (x, y)
            self.target = best_p
            self.target_sid = None

    def _try_rescue(self):
        if self.target_sid is None:
            return
        s = self.env.survivors[self.target_sid]
        if s.state != 'yellow':
            return
        dist = np.sqrt((s.x - self.pos[0])**2 + (s.y - self.pos[1])**2)
        if dist <= self.sensor.rescue_radius:
            result = self.env.try_rescue(
                self.id, *self.pos, self.sensor.rescue_radius)
            if result:
                self.rescued_count += 1
                if self.use_tcfm and self.tcfm:
                    if result.state == 'green':
                        self.tcfm.update_green(
                            result.id, self.id, self.env.current_step)
                    else:
                        self.tcfm.update_red(
                            result.id, self.id, self.env.current_step)
                elif self.broadcast:
                    self.broadcast.local_update(
                        self.id, result.id, result.state)
                self.target = None
                self.target_sid = None
                if self.task_queue and self.task_queue[0] == result.id:
                    self.task_queue.pop(0)
