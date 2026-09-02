"""Learned planner that CAN see the on-route map centerline (privileged).

Fills the one gap in the ability ladder: every other map-using agent is
hand-written. This one consumes the same centerline features but regresses the
trajectory with a small MLP, so the comparison isolates "hand rule vs learned"
with the feature source held fixed.

Reads Scene.map_api + roadblock_ids -> NOT deployable, NOT leaderboard-legal.
Trained by train_map_mlp.py; architecture (hidden / dropout) is read from the
checkpoint so one agent class serves every variant.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.centerline_util import centerline_features, ego_features_from_status
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class _MapMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_mlp(checkpoint_path: str, norm_path: str):
    norm = json.load(open(norm_path, "r", encoding="utf-8"))
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = _MapMLP(int(ckpt["in_dim"]), int(ckpt.get("hidden", 512)),
                    int(ckpt.get("out_dim", 24)), float(ckpt.get("dropout", 0.0)))
    sd = ckpt["state_dict"]
    # Backwards compat: the first checkpoint was saved without Dropout modules
    # (keys net.0/2/4/6). Remap to the dropout-bearing layout (net.0/3/6/9).
    if "net.2.weight" in sd and "net.3.weight" not in sd:
        remap = {"net.0": "net.0", "net.2": "net.3", "net.4": "net.6", "net.6": "net.9"}
        sd = {remap[k.rsplit(".", 1)[0]] + "." + k.rsplit(".", 1)[1]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, np.asarray(norm["mu"], dtype=np.float32), np.asarray(norm["sd"], dtype=np.float32), norm


class MapMLPAgent(AbstractAgent):
    """Learned centerline-conditioned planner (privileged, not deployable)."""

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

    def _fallback(self, ego: np.ndarray) -> Trajectory:
        # No route: straight constant-velocity roll-out rather than an invented path.
        n = self._trajectory_sampling.num_poses
        dt = self._trajectory_sampling.interval_length
        v = float(np.hypot(ego[0], ego[1]))
        poses = np.zeros((n, 3), dtype=np.float32)
        poses[:, 0] = v * dt * np.arange(1, n + 1)
        return Trajectory(poses, self._trajectory_sampling)

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        ego = ego_features_from_status(agent_input.ego_statuses[-1])
        cl = centerline_features(scene, self._k, self._lookahead, self._depth)
        if cl is None:
            return self._fallback(ego)
        x = (np.concatenate([ego, cl.reshape(-1)]).astype(np.float32) - self._mu) / self._sd
        with torch.no_grad():
            out = self._model(torch.tensor(x)[None, :]).numpy()[0]
        return Trajectory(out.reshape(-1, 3).astype(np.float32), self._trajectory_sampling)
