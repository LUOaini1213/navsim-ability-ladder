"""Strict live adapter for real-input models; shuffled models are offline controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.centerline_util import centerline_features, ego_features_from_status
from navsim.agents.ladder_features import agent_features, lead_features
from navsim.agents.mechanism_model import MechanismModel
from navsim.common.dataclasses import SensorConfig, Trajectory


class MechanismAgent(AbstractAgent):
    requires_scene = True

    def __init__(self, run_dir=""):
        super().__init__(requires_scene=True)
        self.run_dir = Path(run_dir)
        self.last_diagnostic = None

    def name(self):
        return self.__class__.__name__

    def initialize(self):
        torch.set_flush_denormal(True)
        settings = json.loads((self.run_dir / "settings.json").read_text(encoding="utf-8"))
        if settings["config"] in ("M2", "M5"):
            raise ValueError("Cross-scene shuffle is an offline negative control, not a live agent")
        checkpoint = torch.load(self.run_dir / "model.pt", map_location="cpu")
        self.mode = checkpoint["mode"]
        self.model = MechanismModel(self.mode, checkpoint["hidden"], checkpoint["layers"], checkpoint["object_hidden"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        norm = json.loads((self.run_dir / "normalization.json").read_text(encoding="utf-8"))
        self.mu, self.sd = np.array(norm["context_mu"], np.float32), np.array(norm["context_sd"], np.float32)
        self.om, self.os = np.array(norm["object_mu"], np.float32), np.array(norm["object_sd"], np.float32)

    def get_sensor_config(self):
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input, scene):
        try:
            frame = scene.frames[scene.scene_metadata.num_history_frames-1]
            ego = ego_features_from_status(agent_input.ego_statuses[-1])
            centerline = centerline_features(scene)
            if centerline is None:
                raise ValueError("Missing required map centerline")
            lead = lead_features(scene, float(np.linalg.norm(ego[:2]))) if self.mode == "lead" else np.zeros(2, np.float32)
            context = (np.concatenate([ego, centerline.reshape(-1), lead])-self.mu)/self.sd
            objects = agent_features(frame) if self.mode in ("concat", "set") else np.zeros((8, 11), np.float32)
            objects[:, :10] = ((objects[:, :10]-self.om)/self.os)*objects[:, 10:11]
            with torch.no_grad():
                poses = self.model(torch.from_numpy(context)[None], torch.from_numpy(objects)[None]).numpy().reshape(8, 3)
            if not np.isfinite(poses).all():
                raise FloatingPointError("Nonfinite poses")
            self.last_diagnostic = {"reason": "model", "used_fallback": False}
            return Trajectory(poses, TrajectorySampling(num_poses=8, interval_length=.5))
        except Exception as error:
            self.last_diagnostic = {"reason": "exception", "used_fallback": False, "exception": repr(error)}
            raise
