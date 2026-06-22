import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISTILLATION_DIR = PROJECT_ROOT / "distillation"
if str(DISTILLATION_DIR) not in sys.path:
    sys.path.insert(0, str(DISTILLATION_DIR))

from distill_trainer import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    DistillationArgs,
    DistillationTrainer,
    PACKED_HISTORY_FORMAT,
    _pack_history_entries,
    _unpack_history_entries,
)


def _example(seed, source=0):
    rng = np.random.default_rng(seed)
    board = rng.integers(-1, 2, size=(6, 5, 5), dtype=np.int8)
    policy = rng.random(150, dtype=np.float32)
    policy /= policy.sum()
    return board, policy, float(rng.uniform(-1.0, 1.0)), source


class PackedHistoryTests(unittest.TestCase):
    def test_round_trip_preserves_history(self):
        history = [
            {"iteration": 3, "examples": [_example(1, 0), _example(2, 4)]},
            {"iteration": 4, "examples": [_example(3, 2)]},
        ]

        packed = _pack_history_entries(history)
        restored = _unpack_history_entries(packed)

        self.assertEqual(packed["format"], PACKED_HISTORY_FORMAT)
        self.assertEqual([entry["iteration"] for entry in restored], [3, 4])
        self.assertEqual([len(entry["examples"]) for entry in restored], [2, 1])
        for expected_entry, restored_entry in zip(history, restored):
            for expected, actual in zip(expected_entry["examples"], restored_entry["examples"]):
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=0.0)
                self.assertAlmostEqual(actual[2], expected[2], places=6)
                self.assertEqual(actual[3], expected[3])
                self.assertEqual(actual[0].dtype, np.int8)
                self.assertEqual(actual[1].dtype, np.float32)

    def test_legacy_history_passes_through(self):
        legacy = [{"iteration": 7, "examples": [_example(7, 1)]}]
        restored = _unpack_history_entries(legacy)
        self.assertEqual(restored, legacy)

    def test_invalid_history_is_rejected(self):
        board, policy, value, source = _example(8)
        policy[0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite policy"):
            _pack_history_entries([{"iteration": 1, "examples": [(board, policy, value, source)]}])

    def test_packed_file_is_materially_smaller(self):
        examples = [_example(index) for index in range(2000)]
        legacy = [{"iteration": 1, "examples": examples}]
        packed = _pack_history_entries(legacy)
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = Path(temp_dir) / "legacy.pth.tar"
            packed_path = Path(temp_dir) / "packed.pth.tar"
            torch.save(legacy, legacy_path)
            torch.save(packed, packed_path)
            self.assertLess(packed_path.stat().st_size, legacy_path.stat().st_size * 0.75)


class CheckpointIntegrationTests(unittest.TestCase):
    def _args(self, root: Path) -> DistillationArgs:
        args = DistillationArgs()
        args.run_name = "checkpoint_test"
        args.checkpoint_root = str(root)
        args.active_model_preset = "fast"
        args.fast_model_preset = {"num_channels": 8, "num_res_blocks": 1}
        args.train_device = "cpu"
        args.shared_inference_device = "cpu"
        args.teacher_model_path = ""
        args.v21_high_model_path = ""
        args.teacher_data_cache_path = str(root / "missing_teacher_cache.pth.tar")
        args.teacher_data_generation_enabled = False
        args.force_overwrite = True
        args.learning_rate_schedule = []
        args.history_window_len = 3
        args.self_play_history_window_len = 3
        args.adversarial_history_window_len = 3
        args.teacher_history_window_len = 3
        return args

    def test_full_checkpoint_resume_and_rng_restore(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trainer = DistillationTrainer(self._args(root))
            trainer.self_play_examples_history = [
                {"iteration": 5, "examples": [_example(10, 0), _example(11, 0)]}
            ]
            trainer.adversarial_examples_history = [
                {"iteration": 5, "examples": [_example(12, 2)]}
            ]
            trainer.eval_history = [{"iteration": 4, "improved": True}]
            trainer.no_refresh_streak = 2
            trainer.recovery_until_iteration = 9
            trainer.optimizer.zero_grad(set_to_none=True)
            optimizer_probe_loss = sum(parameter.square().mean() for parameter in trainer.student.parameters())
            optimizer_probe_loss.backward()
            trainer.optimizer.step()
            trainer.scheduler.step()

            random.seed(123)
            np.random.seed(456)
            torch.manual_seed(789)
            trainer.save_checkpoint(5)
            expected_random = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = float(torch.rand(()))

            checkpoint_path = trainer._checkpoint_path_for_iteration(5)
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["checkpoint_format_version"], CHECKPOINT_FORMAT_VERSION)
            self.assertEqual(payload["self_play_examples_history"]["format"], PACKED_HISTORY_FORMAT)

            resume_args = self._args(root)
            resume_args.force_overwrite = False
            resume_args.resume_path = str(checkpoint_path)
            resumed = DistillationTrainer(resume_args)

            self.assertEqual(resumed.start_iteration, 6)
            self.assertEqual(resumed.last_checkpoint_iteration, 5)
            self.assertEqual(len(resumed.self_play_examples_history), 1)
            self.assertEqual(len(resumed.self_play_examples_history[0]["examples"]), 2)
            self.assertEqual(len(resumed.adversarial_examples_history), 1)
            self.assertEqual(resumed.eval_history, trainer.eval_history)
            self.assertEqual(resumed.no_refresh_streak, 2)
            self.assertEqual(resumed.recovery_until_iteration, 9)
            for key, expected_tensor in trainer.student.state_dict().items():
                torch.testing.assert_close(resumed.student.state_dict()[key], expected_tensor)
            expected_optimizer = trainer.optimizer.state_dict()
            actual_optimizer = resumed.optimizer.state_dict()
            self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
            self.assertEqual(actual_optimizer["state"].keys(), expected_optimizer["state"].keys())
            for state_id, expected_state in expected_optimizer["state"].items():
                for state_key, expected_value in expected_state.items():
                    actual_value = actual_optimizer["state"][state_id][state_key]
                    if torch.is_tensor(expected_value):
                        torch.testing.assert_close(actual_value, expected_value)
                    else:
                        self.assertEqual(actual_value, expected_value)
            self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertEqual(float(torch.rand(())), expected_torch)
            self.assertEqual(resumed.grad_scaler.state_dict(), trainer.grad_scaler.state_dict())

    def test_legacy_checkpoint_is_migrated_on_next_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trainer = DistillationTrainer(self._args(root))
            trainer.self_play_examples_history = [
                {"iteration": 7, "examples": [_example(20, 0), _example(21, 0)]}
            ]
            legacy_state = trainer._build_checkpoint_state(7)
            legacy_state.pop("checkpoint_format_version")
            legacy_state.pop("rng_state")
            legacy_state.pop("grad_scaler")
            legacy_state["self_play_examples_history"] = list(trainer.self_play_examples_history)
            legacy_state["adversarial_examples_history"] = []
            legacy_state["teacher_pure_examples_history"] = []
            legacy_path = root / "legacy_checkpoint.pth.tar"
            torch.save(legacy_state, legacy_path)

            resume_args = self._args(root)
            resume_args.force_overwrite = False
            resume_args.resume_path = str(legacy_path)
            resumed = DistillationTrainer(resume_args)
            self.assertEqual(resumed.start_iteration, 8)
            self.assertEqual(len(resumed.self_play_examples_history[0]["examples"]), 2)

            resumed.save_checkpoint(8)
            migrated = torch.load(
                resumed._checkpoint_path_for_iteration(8),
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(migrated["checkpoint_format_version"], CHECKPOINT_FORMAT_VERSION)
            self.assertEqual(migrated["self_play_examples_history"]["format"], PACKED_HISTORY_FORMAT)

    def test_atomic_save_failure_preserves_existing_target(self):
        trainer = object.__new__(DistillationTrainer)
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "checkpoint.pth.tar"
            target.write_bytes(b"known-good")
            with mock.patch("distill_trainer.torch.save", side_effect=RuntimeError("injected failure")):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    trainer._atomic_torch_save({"new": True}, target)
            self.assertEqual(target.read_bytes(), b"known-good")
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_natural_completion_only_keeps_periodic_saves(self):
        trainer = object.__new__(DistillationTrainer)
        trainer.start_iteration = 1
        trainer.last_checkpoint_iteration = 0
        trainer.args = SimpleNamespace(
            num_iterations=3,
            num_self_play_games=1,
            eval_interval=100,
            checkpoint_interval=2,
        )
        trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3}])
        trainer.scheduler = None
        trainer.iteration_metrics_history = []
        trainer._apply_learning_rate_for_iteration = mock.Mock()
        trainer._is_hot_start_iteration = mock.Mock(return_value=True)
        trainer._build_hot_start_training_examples = mock.Mock(return_value=([], {"mode": "test"}))
        trainer._summarize_runtime_games = mock.Mock(return_value={})
        trainer.train_network = mock.Mock(return_value={})
        trainer._maybe_run_speed_check = mock.Mock(return_value=None)
        trainer._compact_iteration_metrics_history = mock.Mock()
        trainer._log_iteration = mock.Mock()
        trainer.save_checkpoint = mock.Mock()
        trainer._write_final_report = mock.Mock()

        trainer.train()

        trainer.save_checkpoint.assert_called_once_with(2)
        trainer._write_final_report.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
