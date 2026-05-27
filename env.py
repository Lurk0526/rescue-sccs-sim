import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple


@dataclass
class Survivor:
    id: int
    x: int
    y: int
    life_intensity: float
    state: str = 'yellow'
    discovery_step: int = -1
    rescue_step: int = -1
    rescued_by: int = -1


class RescueEnvironment:
    def __init__(self, width=60, height=60, obstacle_ratio=0.25,
                 n_survivors=30, n_clusters=4, cluster_std=6.0,
                 life_decay_rate=0.0003, seed=42):
        self.width = width
        self.height = height
        self.life_decay_rate = life_decay_rate
        self.seed = seed
        self.current_step = 0
        np.random.seed(seed)
        self.grid = self._gen_map(obstacle_ratio)
        self.survivors = self._gen_survivors(n_survivors, n_clusters, cluster_std)
        self.visit_map = np.zeros((height, width), dtype=int)

    def _gen_map(self, ratio):
        g = np.zeros((self.height, self.width), dtype=int)
        target = int(self.width * self.height * ratio)
        placed = 0
        while placed < target:
            x = np.random.randint(1, self.width - 6)
            y = np.random.randint(1, self.height - 6)
            w = np.random.randint(2, 6)
            h = np.random.randint(2, 6)
            g[y:y + h, x:x + w] = 1
            placed += w * h
        g[0, :]  = 1
        g[-1, :] = 1
        g[:, 0]  = 1
        g[:, -1] = 1
        q  = min(self.width, self.height) // 4
        cx = self.width  // 2
        cy = self.height // 2
        r  = min(self.width, self.height) // 3
        fixed = [
            (q,      q),
            (q,      3 * q),
            (3 * q,  q),
            (3 * q,  3 * q),
            (cx,     cy),
        ]
        ring = []
        for i in range(4):
            angle = 2 * np.pi * i / 4
            rx = int(np.clip(cx + r * np.cos(angle), 2, self.width  - 3))
            ry = int(np.clip(cy + r * np.sin(angle), 2, self.height - 3))
            ring.append((rx, ry))
        for sx, sy in fixed + ring:
            g[max(1, sy - 2): sy + 3, max(1, sx - 2): sx + 3] = 0
        return g

    def _robot_starts(self, n=5):
        q  = min(self.width, self.height) // 4
        cx = self.width  // 2
        cy = self.height // 2
        r  = min(self.width, self.height) // 3
        base = [
            (q,      q),
            (q,      3 * q),
            (3 * q,  q),
            (3 * q,  3 * q),
            (cx,     cy),
        ]
        if n <= 5:
            return base[:n]
        extra = []
        for i in range(n - 5):
            angle = 2 * np.pi * i / (n - 5)
            x = int(np.clip(cx + r * np.cos(angle), 2, self.width  - 3))
            y = int(np.clip(cy + r * np.sin(angle), 2, self.height - 3))
            extra.append((x, y))
        return base + extra

    def _gen_survivors(self, n, n_clusters, std):
        free = np.argwhere(self.grid == 0)
        idx  = np.random.choice(len(free), n_clusters, replace=False)
        centers = free[idx]
        survivors, sid = [], 0
        counts = [n // n_clusters] * n_clusters
        counts[-1] += n - sum(counts)
        for ci, (cy, cx) in enumerate(centers):
            placed, attempts = 0, 0
            while placed < counts[ci] and attempts < 2000:
                nx = int(np.clip(cx + np.random.randn() * std, 1, self.width  - 2))
                ny = int(np.clip(cy + np.random.randn() * std, 1, self.height - 2))
                if self.grid[ny, nx] == 0:
                    dist = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
                    intensity = float(np.clip(
                        max(0.25, 1.0 - dist / (std * 2.5)) + np.random.randn() * 0.1,
                        0.1, 1.0))
                    survivors.append(Survivor(sid, nx, ny, intensity))
                    sid += 1
                    placed += 1
                attempts += 1
        return survivors

    def step(self):
        self.current_step += 1
        for s in self.survivors:
            if s.state == 'yellow':
                s.life_intensity = max(0.0, s.life_intensity - self.life_decay_rate)
                if s.life_intensity <= 0:
                    s.state = 'red'

    def get_survivors_in_range(self, x, y, radius,
                               noise_std=0.0, detect_prob=1.0):
        result = []
        for s in self.survivors:
            if s.state != 'yellow':
                continue
            d = np.sqrt((s.x - x) ** 2 + (s.y - y) ** 2)
            if noise_std > 0:
                d += np.random.randn() * noise_std
            if d <= radius and np.random.rand() < detect_prob:
                result.append(s)
        return result

    def get_all_markers_in_range(self, x, y, radius):
        return [s for s in self.survivors
                if np.sqrt((s.x - x) ** 2 + (s.y - y) ** 2) <= radius]

    def try_rescue(self, robot_id, x, y, radius=1.5):
        for s in self.survivors:
            if s.state != 'yellow':
                continue
            if np.sqrt((s.x - x) ** 2 + (s.y - y) ** 2) <= radius:
                s.rescue_step = self.current_step
                s.rescued_by  = robot_id
                s.state = 'green' if s.life_intensity > 0.05 else 'red'
                return s
        return None

    def mark_discovered(self, sid):
        s = self.survivors[sid]
        if s.discovery_step == -1:
            s.discovery_step = self.current_step

    def is_free(self, x, y):
        return (0 <= x < self.width and
                0 <= y < self.height and
                self.grid[y, x] == 0)

    def update_visit(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.visit_map[y, x] += 1

    @property
    def coverage_rate(self):
        total = len(self.survivors)
        if total == 0:
            return 1.0
        return sum(1 for s in self.survivors if s.state == 'green') / total

    @property
    def is_complete(self):
        return all(s.state != 'yellow' for s in self.survivors)

    def get_stats(self):
        rescued = sum(1 for s in self.survivors if s.state == 'green')
        dead    = sum(1 for s in self.survivors if s.state == 'red')
        return {
            'step':          self.current_step,
            'total':         len(self.survivors),
            'rescued':       rescued,
            'dead':          dead,
            'coverage_rate': rescued / len(self.survivors),
        }
