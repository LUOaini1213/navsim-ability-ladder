"""Numerical/behavioral tests; skipped without the optional NAVSIM environment."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
try:
    import numpy as np
    import torch
    from runtime import configure
    configure()
    from navsim.agents import ladder_features as lf
    from navsim.agents.ladder_mlp_agent import LadderMLPAgent
    from navsim.agents.centerline_util import to_ego, to_global
    from navsim.planning.scenario_builder.navsim_scenario_utils import rotate_vector, gt_boxes_oriented_box
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Requires NAVSIM's numpy/torch environment")
class AgentAudit(unittest.TestCase):
    def setUp(self):
        self.status = NS(ego_velocity=np.array([10., 0.]), ego_acceleration=np.zeros(2),
                         driving_command=np.array([0., 1., 0., 0.]))
        self.agent_input = NS(ego_statuses=[self.status])
        self.scene = NS(scene_metadata=NS(num_history_frames=1), frames=[NS(ego_status=self.status)])

    def agent(self):
        agent = LadderMLPAgent()
        agent._see_map, agent._see_agents, agent._target = False, False, "pose"
        agent._mu, agent._sd = np.zeros(8, np.float32), np.ones(8, np.float32)
        agent._model = lambda x: torch.zeros((1, 24))
        return agent

    def test_blind_model_does_not_touch_map_or_annotations(self):
        with patch.object(lf, "route_path", side_effect=AssertionError("map accessed")), \
             patch.object(lf, "centerline_features", side_effect=AssertionError("map accessed")), \
             patch.object(lf, "agent_features", side_effect=AssertionError("annotations accessed")):
            agent = self.agent()
            trajectory = agent.compute_trajectory(self.agent_input, self.scene)
        self.assertEqual(trajectory.poses.shape, (8, 3))
        self.assertEqual(agent.last_diagnostic["reason"], "model")

    def test_agent_input_is_ego_source(self):
        self.scene.frames[0].ego_status = None
        blocks = lf.scene_blocks(self.scene, False, False, False, self.agent_input)
        self.assertEqual(blocks["ego"][0], 10.)

    def test_exception_is_visible_and_strict_can_fail(self):
        agent = self.agent()
        agent._model = lambda x: (_ for _ in ()).throw(ValueError("deliberate"))
        agent.compute_trajectory(self.agent_input, self.scene)
        self.assertEqual(agent.last_diagnostic["exception_type"], "ValueError")
        self.assertTrue(agent.last_diagnostic["used_fallback"])
        agent._strict = True
        with self.assertRaisesRegex(ValueError, "deliberate"):
            agent.compute_trajectory(self.agent_input, self.scene)

    def test_nonfinite_output_is_recorded(self):
        agent = self.agent()
        agent._model = lambda x: torch.full((1, 24), float("nan"))
        agent.compute_trajectory(self.agent_input, self.scene)
        self.assertEqual(agent.last_diagnostic["reason"], "invalid_output")

    def test_boxes_are_sorted_masked_and_range_filtered(self):
        annotations = NS(boxes=np.array([[20, 0, 0, 4, 2, 1, 0], [3, 4, 0, 1, 1, 1, np.pi/2],
                                         [100, 0, 0, 4, 2, 1, 0], [1, 0, 0, 1, 1, 1, 0]]),
                         names=["vehicle", "pedestrian", "vehicle", "traffic_cone"],
                         velocity_3d=np.array([[7, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]]))
        features = lf.agent_features(NS(annotations=annotations))
        np.testing.assert_array_equal(features[:2, :2], [[3, 4], [20, 0]])
        np.testing.assert_array_equal(features[:2, 10], [1, 1])
        np.testing.assert_array_equal(features[2:], 0)
        self.assertAlmostEqual(features[0, 7], 1.)

    def test_known_rotation_matches_upstream(self):
        ego = np.array([100., 200., np.pi/2])
        local = np.array([[10., 2., .3]])
        expected = np.array([[98., 210., np.pi/2 + .3]])
        np.testing.assert_allclose(to_global(local, ego), expected, atol=1e-10)
        np.testing.assert_allclose(to_ego(expected, ego), local, atol=1e-10)
        boxes = np.array([[10., 2., 0., 4., 2., 1., .3]])
        box = gt_boxes_oriented_box(boxes, NS(rear_axle=StateSE2(*ego)))[0]
        np.testing.assert_allclose([box.center.x, box.center.y], expected[0, :2])
        velocity = rotate_vector(StateVector2D(3., 0.), ego[2])
        np.testing.assert_allclose([velocity.x, velocity.y], [0., 3.], atol=1e-10)
        stationary = rotate_vector(StateVector2D(0., 0.), ego[2])
        self.assertEqual(stationary.magnitude(), 0.)

    def test_retiming_endpoint_clipping_is_not_independence(self):
        short = np.array([[1., 0., 0.], [2., 0., 0.]])
        result = lf.retime_polyline(short, np.array([1., 5.]))
        np.testing.assert_array_equal(result[:, 0], [1., 2.])
