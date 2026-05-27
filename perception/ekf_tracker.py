"""
perception/ekf_tracker.py
EKF 时空一致性校验（Level 2 抗干扰机制）

对应论文 3.3.2节：
"结合机器人里程计信息，通过扩展卡尔曼滤波（EKF）
 跟踪标记在全局地图中的时空轨迹。
 若某标记在连续τ帧（本文设τ=3）内的观测残差低于阈值，
 则确认为有效目标；否则标记为候选态。"

状态向量：[x, y, vx, vy]
  x,y   : 标记在全局坐标系中的位置（格）
  vx,vy : 标记速度（废墟中标记静止，用于滤除运动噪声）
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MarkerTrack:
    """单个候选标记的EKF跟踪状态"""
    marker_id:    int
    state:        np.ndarray          # [x, y, vx, vy]
    P:            np.ndarray          # 协方差矩阵 4×4
    color:        str                 # 'yellow'/'green'/'red'（视觉检测结果）
    confirm_count: int = 0            # 连续确认帧数
    total_obs:    int = 0             # 总观测次数
    confirmed:    bool = False        # 是否通过τ帧确认
    residuals:    List[float] = field(default_factory=list)  # 观测残差历史


class EKFTracker:
    """
    扩展卡尔曼滤波器：用于TCFM标记的时空轨迹跟踪

    核心参数（论文对应）：
      tau = 3        ：连续确认帧数阈值
      residual_thresh：观测残差阈值（格），超过则视为噪声

    噪声矩阵设置依据：
      Q（过程噪声）：标记物理固定，运动噪声极小
      R（观测噪声）：YOLOv5检测+AIE后的定位误差约1.5格
    """

    def __init__(self, tau: int = 3, residual_thresh: float = 2.5,
                 dt: float = 1.0):
        self.tau = tau
        self.residual_thresh = residual_thresh
        self.dt = dt                 # 时间步长（仿真步）

        # 状态转移矩阵（匀速运动模型，静止标记vx=vy≈0）
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        # 观测矩阵（只观测位置，不直接观测速度）
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # 过程噪声（标记静止，运动分量极小）
        self.Q = np.diag([0.01, 0.01, 0.001, 0.001])

        # 观测噪声（YOLOv5定位误差：约1.5格）
        self.R = np.diag([2.25, 2.25])

        # 追踪器字典：marker_id → MarkerTrack
        self._tracks: Dict[int, MarkerTrack] = {}
        self._next_id = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        主更新接口，每仿真步调用一次

        detections: YOLOv5检测结果列表
          每个元素: {'x': float, 'y': float, 'color': str, 'conf': float}

        返回：通过τ帧确认的标记列表（可信检测结果）
        """
        # Step 1: 预测（所有现有轨迹向前传播）
        for track in self._tracks.values():
            if not track.confirmed:
                track.state = self.F @ track.state
                track.P = self.F @ track.P @ self.F.T + self.Q

        # Step 2: 数据关联 + 观测更新
        matched_track_ids = set()
        for det in detections:
            obs = np.array([det['x'], det['y']])
            best_id, best_dist = self._associate(obs)

            if best_id is not None:
                # 更新已有轨迹
                self._ekf_update(best_id, obs)
                matched_track_ids.add(best_id)
            else:
                # 新建轨迹
                new_id = self._init_track(obs, det['color'])
                matched_track_ids.add(new_id)

        # Step 3: 未匹配轨迹重置确认计数
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids and not track.confirmed:
                track.confirm_count = max(0, track.confirm_count - 1)

        # Step 4: 返回已确认的标记
        confirmed = []
        for track in self._tracks.values():
            if track.confirmed:
                confirmed.append({
                    'x':     float(track.state[0]),
                    'y':     float(track.state[1]),
                    'color': track.color,
                    'track_id': track.marker_id,
                })
        return confirmed

    def _associate(self, obs: np.ndarray,
                   gate_dist: float = 5.0) -> Tuple[Optional[int], float]:
        """最近邻数据关联（距离门控）

        修复：已确认轨迹也参与关联，避免同一目标被重复建立新轨迹。
        已确认轨迹使用更宽松的门控距离（gate_dist * 1.5），
        确保稳定目标优先匹配，不影响新目标的发现。
        """
        best_id, best_dist = None, gate_dist
        for tid, track in self._tracks.items():
            predicted_pos = track.state[:2]
            dist = float(np.linalg.norm(obs - predicted_pos))
            effective_gate = gate_dist * 1.5 if track.confirmed else gate_dist
            if dist < effective_gate and dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id, best_dist

    def _ekf_update(self, track_id: int, obs: np.ndarray):
        """EKF观测更新步骤"""
        track = self._tracks[track_id]
        # 计算卡尔曼增益
        S = self.H @ track.P @ self.H.T + self.R
        K = track.P @ self.H.T @ np.linalg.inv(S)
        # 状态更新
        innovation = obs - self.H @ track.state
        residual = float(np.linalg.norm(innovation))
        track.state = track.state + K @ innovation
        track.P = (np.eye(4) - K @ self.H) @ track.P
        track.residuals.append(residual)
        track.total_obs += 1

        # 判断是否通过时空一致性（残差 < 阈值 且 连续τ帧）
        if residual < self.residual_thresh:
            track.confirm_count += 1
        else:
            track.confirm_count = max(0, track.confirm_count - 1)

        if track.confirm_count >= self.tau:
            track.confirmed = True

    def _init_track(self, obs: np.ndarray, color: str) -> int:
        """初始化新轨迹"""
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = MarkerTrack(
            marker_id=tid,
            state=np.array([obs[0], obs[1], 0.0, 0.0]),
            P=np.eye(4) * 5.0,
            color=color,
        )
        return tid

    def reset(self):
        self._tracks.clear()
        self._next_id = 0

    def get_stats(self) -> Dict:
        confirmed = sum(1 for t in self._tracks.values() if t.confirmed)
        pending   = sum(1 for t in self._tracks.values() if not t.confirmed)
        return {'confirmed': confirmed, 'pending': pending,
                'total_tracks': len(self._tracks)}
