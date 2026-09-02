"""Hand-written path + learned speed (privileged).

The map centerline supplies the geometry; a small MLP supplies only the
progress along it (8 arc-length offsets). Tests whether a learned model can
capture the one thing the ladder says is worth the most — the speed profile —
once it is no longer asked to draw the line as well.

Not deployable (reads Scene.map_api). Trained by train_speed_mlp.py.
"""

from __future__ import annotations

import numpy as np
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.centerline_util import (centerline_features, ego_features_from_status,
                                           route_path, to_ego)
from navsim.agents.map_mlp_agent import load_mlp
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class SpeedMLPAgent(AbstractAgent):
    requires_scene = True

    def __init__(self, trajectory_sampling: TrajectorySampling, checkpoint_path: str, norm_path: str):
        super().__init__(requires_scene=True)
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path
        self._norm_path = norm_path
        self._model = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        self._model, self._mu, self._sd, norm = load_mlp(self._checkpoint_path, self._norm_path)
        self._k = int(norm.get("k_points", 20))
        self._lookahead = float(norm.get("lookahead_m", 60.0))
        self._depth = int(norm.get("search_depth", 15))

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def _fallback(self, ego_feat: np.ndarray) -> Trajectory:
        n = self._trajectory_sampling.num_poses
        dt = self._trajectory_sampling.interval_length
        v = float(np.hypot(ego_feat[0], ego_feat[1]))
        poses = np.zeros((n, 3), dtype=np.float32)
        poses[:, 0] = v * dt * np.arange(1, n + 1)
        return Trajectory(poses, self._trajectory_sampling)

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        ego_feat = ego_features_from_status(agent_input.ego_statuses[-1])
        rp = route_path(scene, self._depth)
        cl = centerline_features(scene, self._k, self._lookahead, self._depth)
        if rp is None or cl is None:
            return self._fallback(ego_feat)
        path, ego, s0 = rp
        x = (np.concatenate([ego_feat, cl.reshape(-1)]).astype(np.float32) - self._mu) / self._sd
        with torch.no_grad():
            ds = self._model(torch.tensor(x)[None, :]).numpy()[0].astype(np.float64)
        ds = np.maximum.accumulate(np.clip(ds, 0.0, None))
        try:
            g = np.asarray(path.interpolate(s0 + ds, as_array=True), dtype=np.float64)
        except Exception:
            return self._fallback(ego_feat)
        return Trajectory(to_ego(g, ego).astype(np.float32), self._trajectory_sampling)
