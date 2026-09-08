"""One learned agent class for every cell of the ladder / factorial / path-speed tables.

The training run directory (scripts/train_ladder.py) holds model.pt + norm.json;
norm.json records which feature blocks the model saw (see_map, see_agents) and
what it predicts (target = "pose": 8 ego-frame poses, or "ds": 8 arc-length
offsets along the rule centerline path).

speed_mode (config override) only matters for target="pose":
  learned  use the predicted poses as-is                (learned path + learned speed)
  rule     keep the predicted geometry, re-time it with the kinematic
           constant-acceleration law                     (learned path + rule speed)

target="ds" is always rule path + learned speed. Privileged (reads Scene.map_api
and GT boxes) -> not deployable, not leaderboard-legal.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.kinematic_agent import KinematicAgent
from navsim.agents.ladder_features import assemble, kinematic_progress, poses_from_progress, retime_polyline, scene_blocks
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class LadderMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float, layers: int = 3):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            d = hidden
        mods.append(torch.nn.Linear(d, out_dim))
        self.net = torch.nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def load_run(run_dir: str):
    torch.set_flush_denormal(True)  # regularised checkpoints hold subnormal weights; avoid the CPU slow path
    norm = json.load(open(f"{run_dir}/norm.json", "r", encoding="utf-8"))
    ckpt = torch.load(f"{run_dir}/model.pt", map_location="cpu")
    model = LadderMLP(int(ckpt["in_dim"]), int(ckpt["hidden"]), int(ckpt["out_dim"]), float(ckpt["dropout"]), int(ckpt.get("layers", 3)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, np.asarray(norm["mu"], np.float32), np.asarray(norm["sd"], np.float32), norm


class LadderMLPAgent(AbstractAgent):
    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        run_dir: str = "",
        speed_mode: str = "learned",
    ):
        super().__init__(requires_scene=True)
        self._trajectory_sampling = trajectory_sampling
        self._run_dir = run_dir
        self._speed_mode = speed_mode
        self._model = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        self._model, self._mu, self._sd, self._norm = load_run(self._run_dir)
        self._see_map = bool(self._norm["see_map"])
        self._see_agents = bool(self._norm["see_agents"])
        self._target = str(self._norm["target"])

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def _fallback(self, agent_input: AgentInput) -> Trajectory:
        return KinematicAgent(trajectory_sampling=self._trajectory_sampling).compute_trajectory(agent_input)

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        try:
            blocks = scene_blocks(scene)
            if blocks is None:
                return self._fallback(agent_input)
            x = assemble(blocks["ego"], blocks["cl"], blocks["agents"], blocks["lead"], self._see_map, self._see_agents)
            x = (x - self._mu) / self._sd
            with torch.no_grad():
                out = self._model(torch.tensor(x)[None, :]).numpy()[0].astype(np.float64)
            n, dt = self._trajectory_sampling.num_poses, float(self._trajectory_sampling.interval_length)
            if self._target == "ds":
                poses = poses_from_progress(blocks["path"], blocks["ego_pose"], blocks["s0"], out[:n])
            else:
                poses = out.reshape(-1, 3)[:n].astype(np.float32)
                if self._speed_mode == "rule":
                    a_long = float(blocks["ego"][2])
                    poses = retime_polyline(poses, kinematic_progress(blocks["speed"], a_long, n, dt))
            if poses.shape == (n, 3) and np.all(np.isfinite(poses)):
                return Trajectory(poses.astype(np.float32), self._trajectory_sampling)
        except Exception:
            pass
        return self._fallback(agent_input)
