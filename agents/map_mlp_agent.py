"""Learned planner that CAN see the on-route map centerline.

Fills the one gap in the ability ladder: every other map-using agent
(PrivMap / PrivMapKin / PrivMapGTSpd) is hand-written. This one consumes the
same centerline features but regresses the trajectory with a small MLP, so the
comparison isolates "hand rule vs learned" with the feature source held fixed.

Privileged: it reads Scene.map_api + roadblock_ids, so it is NOT deployable and
NOT leaderboard-legal. Trained by train_map_mlp.py on the 51 train logs of
available_mini_logs.yaml; evaluate on the 13 val logs for a clean number.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from shapely.geometry import Point

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


class _MapMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 24):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MapMLPAgent(AbstractAgent):
    """Learned centerline-conditioned planner (privileged, not deployable)."""

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        checkpoint_path: str,
        norm_path: str,
    ):
        super().__init__(requires_scene=True)
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path
        self._norm_path = norm_path
        self._model = None
        self._mu = None
        self._sd = None
        self._k = 20
        self._lookahead = 60.0
        self._depth = 15

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        norm = json.load(open(self._norm_path, "r", encoding="utf-8"))
        self._mu = np.asarray(norm["mu"], dtype=np.float32)
        self._sd = np.asarray(norm["sd"], dtype=np.float32)
        self._k = int(norm.get("k_points", 20))
        self._lookahead = float(norm.get("lookahead_m", 60.0))
        self._depth = int(norm.get("search_depth", 15))
        ckpt = torch.load(self._checkpoint_path, map_location="cpu")
        self._model = _MapMLP(int(ckpt["in_dim"]))
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def _centerline(self, scene: Scene):
        frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        ids = list(dict.fromkeys(frame.roadblock_ids))
        blocks, lanes = {}, {}
        for rid in ids:
            b = scene.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK)
            b = b or scene.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if b is None:
                continue
            blocks[b.id] = b
            for lane in b.interior_edges:
                lanes[lane.id] = lane
        if not lanes:
            return None
        ego = np.asarray(frame.ego_status.ego_pose, dtype=np.float64)
        ego_pt = Point(float(ego[0]), float(ego[1]))
        best, best_d = None, 1e9
        for lane in lanes.values():
            try:
                d = float(lane.baseline_path.linestring.distance(ego_pt))
            except Exception:
                continue
            if d < best_d:
                best_d, best = d, lane
        if best is None:
            return None
        block_ids = list(blocks.keys())
        block_list = list(blocks.values())
        try:
            start_idx = int(np.argmax(np.array(block_ids) == best.get_roadblock_id()))
            window = block_list[start_idx : start_idx + self._depth] or block_list[: self._depth]
            route_plan, _ = Dijkstra(best, list(lanes.keys())).search(window[-1])
            discrete = []
            for lane in route_plan:
                discrete.extend(lane.baseline_path.discrete_path)
        except Exception:
            discrete = []
        if len(discrete) < 2:
            discrete = list(best.baseline_path.discrete_path)
        if len(discrete) < 2:
            return None
        path = PDMPath(discrete)
        s0 = float(path.project(ego_pt))
        ss = s0 + np.linspace(0.0, self._lookahead, self._k)
        try:
            poses = path.interpolate(ss, as_array=True)
        except Exception:
            return None
        yaw = float(ego[2])
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s], [s, c]])
        rel = (poses[:, :2] - ego[:2]) @ R.T
        dth = poses[:, 2] - yaw
        return np.concatenate(
            [rel, np.cos(dth)[:, None], np.sin(dth)[:, None]], axis=1
        ).astype(np.float32)

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        cl = self._centerline(scene)
        status = agent_input.ego_statuses[-1]
        ego = np.concatenate(
            [
                np.asarray(status.ego_velocity, dtype=np.float32),
                np.asarray(status.ego_acceleration, dtype=np.float32),
                np.asarray(status.driving_command, dtype=np.float32),
            ]
        ).astype(np.float32)

        if cl is None:
            # No route available: fall back to a straight constant-velocity roll-out
            # rather than inventing a path. Reported honestly in the eval notes.
            n = self._trajectory_sampling.num_poses
            dt = self._trajectory_sampling.interval_length
            v = float(np.hypot(ego[0], ego[1]))
            poses = np.zeros((n, 3), dtype=np.float32)
            poses[:, 0] = v * dt * np.arange(1, n + 1)
            return Trajectory(poses, self._trajectory_sampling)

        x = np.concatenate([ego, cl.reshape(-1)]).astype(np.float32)
        x = (x - self._mu) / self._sd
        with torch.no_grad():
            out = self._model(torch.tensor(x)[None, :]).numpy()[0]
        poses = out.reshape(-1, 3).astype(np.float32)
        return Trajectory(poses, self._trajectory_sampling)
