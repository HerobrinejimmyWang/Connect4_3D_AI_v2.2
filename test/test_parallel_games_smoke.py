import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from model import Connect4Net, extract_model_config  # noqa: E402
from parallel_games import execute_evaluation_group_parallel, execute_self_play_parallel  # noqa: E402


class ParallelGamesSmokeTests(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            cpuct=1.0,
            num_mcts_sims=1,
            num_mcts_threads=1,
            virtual_loss=1.0,
            reuse_mcts_tree=True,
            persistent_mcts_threads=True,
            enable_mcts_search_stats=True,
            inference_precision="fp32",
            shared_inference_server_count=1,
            compatible_inference_server_count=1,
            high_mcts_shared_inference_server_threshold=0,
            high_mcts_shared_inference_server_count=1,
            self_play_phase_schedule=[
                {
                    "name": "smoke",
                    "max_step": None,
                    "temperature": 1.0,
                    "dirichlet_alpha": 0.0,
                    "dirichlet_epsilon": 0.0,
                }
            ],
            exploration_iteration_schedule=[],
            self_play_exploration_strength=1.0,
            current_iteration=1,
            min_game_steps=1,
            min_game_steps_start_iteration=999999,
            tactical_override_max_step=0,
        )

    def test_cpu_self_play_uses_shared_service_and_reports_search_stats(self):
        model = Connect4Net(board_layers=6, board_size=5, num_channels=4, num_res_blocks=1)
        args = self._args()
        examples, games = execute_self_play_parallel(
            args=args,
            num_games=1,
            num_workers=1,
            shared_inference_device="cpu",
            inference_batch_size=8,
            inference_timeout_s=0.001,
            model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
            model_config=extract_model_config(model),
            progress_desc="CPU smoke",
        )
        self.assertEqual(len(games), 1)
        self.assertGreater(games[0]["search_stats"]["simulations"], 0)
        self.assertGreater(len(examples), 0)
        for _, policy, _ in examples:
            self.assertTrue(np.isfinite(policy).all())
            self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5)

    def test_grouped_evaluation_shares_candidate_service(self):
        model = Connect4Net(board_layers=6, board_size=5, num_channels=4, num_res_blocks=1)
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        results = execute_evaluation_group_parallel(
            args=self._args(),
            matches=[
                {"label": "random_a", "num_games": 1},
                {"label": "random_b", "num_games": 1},
            ],
            total_workers=2,
            shared_inference_device="cpu",
            inference_batch_size=8,
            inference_timeout_s=0.001,
            new_model_state=state,
            new_model_config=extract_model_config(model),
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(sum(counts) == 1 for counts in results))


if __name__ == "__main__":
    unittest.main()
