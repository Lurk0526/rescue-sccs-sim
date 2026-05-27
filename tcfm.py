import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Marker:
    survivor_id: int
    x: int; y: int
    color: str          # yellow / green / red
    placed_by: int
    placed_step: int
    read_count: int = 0

class TCFMSystem:
    def __init__(self):
        self._markers: Dict[int, Marker] = {}
        self.marker_reads = 0

    def place_yellow(self, sid, x, y, robot_id, step):
        if sid not in self._markers:
            self._markers[sid] = Marker(sid, x, y, 'yellow', robot_id, step)

    def update_green(self, sid, robot_id, step):
        if sid in self._markers:
            m = self._markers[sid]; m.color='green'; m.placed_by=robot_id

    def update_red(self, sid, robot_id, step):
        if sid in self._markers:
            m = self._markers[sid]; m.color='red'; m.placed_by=robot_id

    def read_in_range(self, x, y, radius, robot_id) -> List[Marker]:
        result = []
        for m in self._markers.values():
            if np.sqrt((m.x-x)**2+(m.y-y)**2) <= radius:
                m.read_count += 1; self.marker_reads += 1
                result.append(m)
        return result

    def is_done(self, sid):
        m = self._markers.get(sid)
        return m is not None and m.color in ('green','red')

    def is_marked(self, sid):
        return sid in self._markers

    def get_stats(self):
        return {
            'total': len(self._markers),
            'yellow': sum(1 for m in self._markers.values() if m.color=='yellow'),
            'green':  sum(1 for m in self._markers.values() if m.color=='green'),
            'red':    sum(1 for m in self._markers.values() if m.color=='red'),
            'reads':  self.marker_reads
        }

class BroadcastComm:
    """对比用：模拟传统广播通信（每N步同步）"""
    def __init__(self, n_robots, sync_interval=15):
        self.n = n_robots
        self.interval = sync_interval
        self.local: List[Dict] = [{} for _ in range(n_robots)]
        self.broadcast_count = 0

    def local_update(self, robot_id, sid, state):
        self.local[robot_id][sid] = state

    def sync(self, step) -> int:
        if step % self.interval != 0:
           return 0
    # 同一步内多个机器人都会调用sync，只计算一次
        if getattr(self, '_last_synced', -1) == step:
           return 0
        self._last_synced = step
        merged = {}
        for d in self.local:
            merged.update(d)
        for i in range(self.n):
            self.local[i].update(merged)
        msgs = self.n * (self.n - 1)
        self.broadcast_count += msgs
        return msgs

    def is_known_done(self, robot_id, sid):
        return self.local[robot_id].get(sid) in ('green','red')
