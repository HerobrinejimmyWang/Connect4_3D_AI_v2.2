import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
DISTILLATION_DIR = PROJECT_ROOT / "distillation"
for path in (TRAINING_DIR, DISTILLATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distill_trainer import DistillationArgs, DistillationDataset, DistillationTrainer  # noqa: E402
from game_rules import GameRules  # noqa: E402
from mcts import MCTS, _encode_board_batch  # noqa: E402
from model import board_to_channels  # noqa: E402
from model_compat import encode_board_batch_for_model, encode_board_for_model  # noqa: E402


class UniformInference:
    def __init__(self, fail_after_root=False):
        self.fail_after_root = fail_after_root
        self.calls = 0

    def submit_and_wait(self, state):
        self.calls += 1
        if self.fail_after_root and np.count_nonzero(state):
            raise RuntimeError("worker inference failure")
        return np.full(150, 1.0 / 150, dtype=np.float32), 0.0


def _args(**overrides):
    values = dict(
        cpuct=1.0,
        num_mcts_sims=8,
        num_mcts_threads=2,
        virtual_loss=1.0,
        inference_batch_size=8,
        inference_timeout_s=0.001,
        reuse_mcts_tree=True,
        persistent_mcts_threads=True,
        enable_mcts_search_stats=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class MCTSLifecycleTests(unittest.TestCase):
    def test_advance_preserves_child_and_runs_exact_new_simulations(self):
        game = GameRules()
        mcts = MCTS(game, UniformInference(), _args())
        try:
            board = game.get_init_board()
            mcts.get_action_prob(board, temp=1)
            child = max(mcts.root.children.values(), key=lambda node: node.visit_count)
            action = int(child.action_from_parent)
            visits = child.visit_count

            self.assertTrue(mcts.advance_to_action(action, child.state.copy()))
            self.assertIs(mcts.root, child)
            self.assertIsNone(mcts.root.parent)
            self.assertEqual(mcts.root.visit_count, visits)

            before = mcts.get_search_stats()["simulations"]
            mcts.get_action_prob(child.state, temp=1)
            self.assertEqual(mcts.get_search_stats()["simulations"] - before, 8)
        finally:
            mcts.close()

    def test_missing_child_and_state_mismatch_reset_safely(self):
        game = GameRules()
        mcts = MCTS(game, UniformInference(), _args(num_mcts_threads=1))
        try:
            board = game.get_init_board()
            mcts.get_action_prob(board)
            self.assertFalse(mcts.advance_to_action(149, np.ones_like(board)))
            self.assertEqual(mcts.get_search_stats()["reset_reasons"]["missing_child"], 1)

            mcts.get_action_prob(board)
            action = next(iter(mcts.root.children))
            self.assertFalse(mcts.advance_to_action(action, np.ones_like(board)))
            self.assertIn("child_state_mismatch", mcts.get_search_stats()["reset_reasons"])
        finally:
            mcts.close()

    def test_legacy_flags_reset_root_and_do_not_keep_executor(self):
        game = GameRules()
        mcts = MCTS(
            game,
            UniformInference(),
            _args(reuse_mcts_tree=False, persistent_mcts_threads=False),
        )
        try:
            board = game.get_init_board()
            mcts.get_action_prob(board)
            first_root = mcts.root
            mcts.get_action_prob(board)
            self.assertIsNot(mcts.root, first_root)
            self.assertIsNone(mcts._executor)
        finally:
            mcts.close()

    def test_persistent_executor_is_reused_and_worker_errors_propagate(self):
        game = GameRules()
        mcts = MCTS(game, UniformInference(), _args(num_mcts_sims=2))
        executor = mcts._executor
        try:
            board = game.get_init_board()
            mcts.get_action_prob(board)
            mcts.get_action_prob(board)
            self.assertIs(mcts._executor, executor)
        finally:
            mcts.close()
        self.assertIsNone(mcts._executor)

        failing = MCTS(game, UniformInference(fail_after_root=True), _args(num_mcts_sims=1))
        try:
            with self.assertRaisesRegex(RuntimeError, "worker inference failure"):
                failing.get_action_prob(game.get_init_board())
        finally:
            failing.close()


class EncodingAndDatasetTests(unittest.TestCase):
    def test_vectorized_encoders_match_scalar_encoding(self):
        rng = np.random.default_rng(42)
        boards = rng.integers(-1, 2, size=(5, 6, 5, 5), dtype=np.int8)
        expected = np.stack([board_to_channels(board) for board in boards])
        np.testing.assert_array_equal(_encode_board_batch(boards), expected)

        config = {"board_layers": 4, "board_size": 4, "input_channels": 2}
        expected_compatible = np.stack([encode_board_for_model(board, config) for board in boards])
        np.testing.assert_array_equal(encode_board_batch_for_model(boards, config), expected_compatible)

    def test_dataset_keeps_compact_raw_boards(self):
        board = np.zeros((6, 5, 5), dtype=np.int8)
        policy = np.full(150, 1.0 / 150, dtype=np.float32)
        dataset = DistillationDataset([(board, policy, 0.5, 0)] * 4)
        self.assertEqual(dataset.boards.dtype, torch.int8)
        self.assertEqual(tuple(dataset.boards.shape), (4, 6, 5, 5))
        old_encoded_bytes = 4 * 2 * 6 * 5 * 5 * 4
        self.assertLess(dataset.boards.numel() * dataset.boards.element_size(), old_encoded_bytes * 0.2)

    def test_cpu_train_save_resume_and_continue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = DistillationArgs()
            args.run_name = "pipeline_smoke"
            args.checkpoint_root = temp_dir
            args.active_model_preset = "fast"
            args.fast_model_preset = {"num_channels": 4, "num_res_blocks": 1}
            args.train_device = "cpu"
            args.shared_inference_device = "cpu"
            args.teacher_model_path = ""
            args.v21_high_model_path = ""
            args.teacher_data_cache_path = str(Path(temp_dir) / "missing.pth.tar")
            args.teacher_data_generation_enabled = False
            args.force_overwrite = True
            args.learning_rate_schedule = []
            args.epochs = 1
            args.batch_size = 2

            policy = np.full(150, 1.0 / 150, dtype=np.float32)
            examples = [
                (np.zeros((6, 5, 5), dtype=np.int8), policy, 0.0, 0),
                (np.ones((6, 5, 5), dtype=np.int8), policy, 1.0, 1),
            ]
            trainer = DistillationTrainer(args)
            metrics = trainer.train_network(examples, 1)
            self.assertGreater(metrics["samples_per_sec"], 0.0)
            self.assertEqual(metrics["dataset_cpu_bytes"], 2 * (150 + 150 * 4 + 4 + 8))
            trainer.save_checkpoint(1)
            checkpoint_path = trainer.run_dir / "checkpoint_1" / "checkpoint.pth.tar"

            resume_args = DistillationArgs.from_dict(args.to_dict())
            resume_args.resume = True
            resume_args.resume_path = str(checkpoint_path)
            resumed = DistillationTrainer(resume_args)
            self.assertEqual(resumed.start_iteration, 2)
            continued = resumed.train_network(examples, 2)
            self.assertTrue(np.isfinite(continued["total_loss"]))


if __name__ == "__main__":
    unittest.main()
