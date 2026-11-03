import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
try:
    import numpy as np
    import torch
    from mechanism_data import (fit_normalizer, transform, matched_donors, session_id,
                                freeze_protocol, split_indices)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))
    from mechanism_model import MechanismModel
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Requires optional numpy/torch")
class MechanismTests(unittest.TestCase):
    def fixture(self, n=40):
        return {"ego": np.arange(n*8, dtype=np.float32).reshape(n, 8),
                "cl": np.ones((n, 80), np.float32), "lead": np.ones((n, 2), np.float32),
                "agents": np.zeros((n, 8, 11), np.float32), "speed": np.ones(n), "cmd": np.ones(n),
                "log_name": np.array([f"session{i//2}_000_001" for i in range(n)]),
                "token": np.array([f"token{i}" for i in range(n)]),
                "map_name": np.array(["city"]*n), "y_pose": np.zeros((n, 24), np.float32)}

    def test_original_session_parser(self):
        self.assertEqual(session_id("2021.06.03.12.02.06_veh-35_01100_01227"), "2021.06.03.12.02.06_veh-35")
        with self.assertRaises(ValueError):
            session_id("unknown")

    def test_masked_normalization_excludes_padding_and_holdout(self):
        data = self.fixture(4)
        data["agents"][:2, 0, :6] = [[2]*6, [4]*6]
        data["agents"][:2, 0, 10] = 1
        data["agents"][2:, 0, :6] = 1e6
        data["agents"][2:, 0, 10] = 1
        norm = fit_normalizer(data, [0, 1])
        self.assertEqual(norm["object_mu"][:6], [3.]*6)
        context, objects = transform(data, [0, 1, 2, 3], norm)
        np.testing.assert_array_equal(objects[:, 1:], 0)
        np.testing.assert_array_equal(objects[:2, 0, 0], [-1, 1])
        self.assertEqual(context.dtype, np.float32)

    def test_controls_are_reproducible_disjoint_and_do_not_read_targets(self):
        data = self.fixture()
        ids = np.arange(20)
        first, levels = matched_donors(data, ids, 123)
        data["y_pose"][:] = float("nan")
        second, _ = matched_donors(data, ids, 123)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(set(first) <= set(ids))
        for recipient, donor in zip(ids, first):
            self.assertNotEqual(session_id(data["log_name"][recipient]), session_id(data["log_name"][donor]))
        self.assertEqual(sum(levels.values()), len(ids))

    def test_frozen_splits_are_session_disjoint_and_nested(self):
        train, test = self.fixture(), self.fixture(4)
        test["log_name"] = np.array(["testsession_000_001"]*4)
        test["token"] = np.array([f"test{i}" for i in range(4)])
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)/"source"
            p.write_bytes(b"fixture")
            protocol = freeze_protocol(train, test, p, p)
        sets = [set(protocol["split_sessions"][key]) for key in ("train", "validation", "confirmation")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        small = set(split_indices(train, protocol, "train", .1))
        medium = set(split_indices(train, protocol, "train", .3))
        full = set(split_indices(train, protocol, "train", 1.))
        self.assertTrue(small <= medium <= full)

    def test_set_encoder_is_invariant_and_empty_padding_is_safe(self):
        torch.manual_seed(0)
        model = MechanismModel("set", hidden=16, layers=2, object_hidden=8)
        context = torch.randn(4, 90)
        objects = torch.randn(4, 8, 11)
        objects[:, :, 10] = 0
        objects[:2, :3, 10] = 1
        original = model(context, objects)
        order = torch.tensor([7, 1, 4, 3, 2, 5, 0, 6])
        torch.testing.assert_close(original, model(context, objects[:, order]), atol=1e-6, rtol=1e-6)
        poisoned = objects.clone()
        poisoned[:, 3:, :10] = 10000
        torch.testing.assert_close(original, model(context, poisoned), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(original).all())
        original.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_unused_branches_cannot_affect_baseline(self):
        model = MechanismModel("none", hidden=16, layers=2)
        context, objects = torch.randn(2, 90), torch.randn(2, 8, 11)
        first = model(context, objects)
        context[:, 88:] += 1000
        torch.testing.assert_close(first, model(context, objects*1000))

    def test_parameter_budget_within_ten_percent(self):
        counts = [sum(p.numel() for p in MechanismModel(mode).parameters())
                  for mode in ("none", "concat", "lead", "set")]
        self.assertLess(max(counts)/min(counts), 1.1)

    def test_checkpoint_restores_optimizer_sampler_and_rng(self):
        from train_mechanism import atomic_torch
        torch.manual_seed(73)
        model = MechanismModel("none", hidden=16, layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        context, objects, target = torch.randn(10, 90), torch.zeros(10, 8, 11), torch.randn(10, 24)
        order = torch.randperm(10)
        def update(net, opt, ids):
            opt.zero_grad(set_to_none=True)
            loss = (net(context[ids], objects[ids])-target[ids]).abs().mean()
            loss.backward()
            opt.step()
        update(model, optimizer, order[:2])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"resume.pt"
            atomic_torch(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "rng": torch.get_rng_state(), "order": order, "position": 2})
            update(model, optimizer, order[2:4])
            expected_order = torch.randperm(10)
            restored = MechanismModel("none", hidden=16, layers=2)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
            state = torch.load(path, map_location="cpu", weights_only=False)
            restored.load_state_dict(state["model"])
            restored_optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            update(restored, restored_optimizer, state["order"][state["position"]:state["position"]+2])
            torch.testing.assert_close(expected_order, torch.randperm(10), atol=0, rtol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)

    def test_confirmation_gate_rejects_missing_core_models(self):
        from predict_mechanism import verify_core
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                verify_core(Path(temporary), "not-a-protocol")
