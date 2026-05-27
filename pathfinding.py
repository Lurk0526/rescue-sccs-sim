import heapq
import numpy as np
from typing import Tuple, List, Dict, Optional

def astar(grid, start: Tuple, goal: Tuple) -> List[Tuple]:
    if start == goal: return []
    H, W = grid.shape
    def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    open_h = [(h(start), start)]
    came, g = {}, {start: 0}
    while open_h:
        _, cur = heapq.heappop(open_h)
        if cur == goal:
            path = []
            while cur in came: path.append(cur); cur = came[cur]
            return list(reversed(path))
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nb = (cur[0]+dx, cur[1]+dy)
            if 0<=nb[0]<W and 0<=nb[1]<H and grid[nb[1],nb[0]]==0:
                ng = g[cur] + 1
                if nb not in g or ng < g[nb]:
                    came[nb] = cur; g[nb] = ng
                    heapq.heappush(open_h, (ng+h(nb), nb))
    return []

def next_step(grid, pos, goal, cache: Dict = None) -> Tuple:
    if pos == goal: return pos
    key = (pos, goal)
    if cache is not None and key in cache:
        path = cache[key]
    else:
        path = astar(grid, pos, goal)
        if cache is not None: cache[key] = path
    if path: return path[0]
    # 无路径时随机相邻空格
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    np.random.shuffle(dirs)
    for dx, dy in dirs:
        nx, ny = pos[0]+dx, pos[1]+dy
        if 0<=nx<grid.shape[1] and 0<=ny<grid.shape[0] and grid[ny,nx]==0:
            return (nx, ny)
    return pos
