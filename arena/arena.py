from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CURRENT_DIR.parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from arena_game_rules import GameRules


class Arena:
    def __init__(
        self,
        agent1,
        agent2,
        game_factory,
        num_games=20,
        parallel_games=1,
        temperature=0,
        max_moves=None,
        immediate_win_check=False,
        observer=None,
    ):
        self.agent1 = agent1
        self.agent2 = agent2
        self.game_factory = game_factory
        self.num_games = int(num_games)
        self.parallel_games = max(1, int(parallel_games))
        self.temperature = float(temperature)
        self.max_moves = max_moves
        self.immediate_win_check = bool(immediate_win_check)
        self.observer = observer

    def play(self):
        fixtures = [(game_index, game_index % 2 == 1) for game_index in range(self.num_games)]
        if self.parallel_games == 1:
            results = [self._play_single_game(game_index, swap_players) for game_index, swap_players in fixtures]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=self.parallel_games) as executor:
                future_map = {
                    executor.submit(self._play_single_game, game_index, swap_players): game_index
                    for game_index, swap_players in fixtures
                }
                for future in as_completed(future_map):
                    results.append(future.result())
            results.sort(key=lambda item: item["game_index"])
        return self._summarize(results)

    def close(self):
        self.agent1.close()
        self.agent2.close()

    def _play_single_game(self, game_index, swap_players):
        game = self.game_factory()
        if swap_players:
            first_agent = self.agent2
            second_agent = self.agent1
        else:
            first_agent = self.agent1
            second_agent = self.agent2

        player_to_agent = {1: first_agent, -1: second_agent}
        board = game.get_init_board()
        current_player = 1
        move_count = 0
        move_history = []
        total_search_time = {
            self.agent1.name: 0.0,
            self.agent2.name: 0.0,
        }
        total_search_count = {
            self.agent1.name: 0,
            self.agent2.name: 0,
        }
        tactical_move_count = {
            self.agent1.name: 0,
            self.agent2.name: 0,
        }

        if self.observer is not None:
            self.observer.on_game_start(game_index, first_agent.name, second_agent.name, board)

        while True:
            acting_agent = player_to_agent[current_player]
            start = time.perf_counter()
            decision_source = "model_search"
            tactical_reason = None
            forced_action = self._select_forced_action(game, board, current_player, acting_agent)
            if forced_action is not None:
                action = int(forced_action["action"])
                decision_source = forced_action["source"]
                tactical_reason = forced_action["label"]
                tactical_move_count[acting_agent.name] += 1
            else:
                action, _ = acting_agent.get_action(board, current_player, temp=self.temperature)
            elapsed = time.perf_counter() - start
            if decision_source == "model_search":
                total_search_time[acting_agent.name] += elapsed
                total_search_count[acting_agent.name] += 1
            move_count += 1

            valid_moves = game.get_valid_moves(board)
            if action < 0 or action >= len(valid_moves) or valid_moves[action] == 0:
                winner_agent = player_to_agent[-current_player].name
                result = {
                    "game_index": game_index,
                    "winner": winner_agent,
                    "draw": False,
                    "illegal_move_by": acting_agent.name,
                    "move_count": move_count,
                    "move_time": total_search_time,
                    "move_decisions": total_search_count,
                    "tactical_moves": tactical_move_count,
                    "first_agent": first_agent.name,
                    "second_agent": second_agent.name,
                    "moves": move_history,
                    "final_board": board.tolist(),
                    "status": "illegal_move",
                }
                if self.observer is not None:
                    self.observer.on_game_complete(game_index, result)
                return result

            board, current_player = game.get_next_state(board, current_player, action)
            layer, row, col = game.action_to_coords(action)
            move_record = {
                "move_number": move_count,
                "player": -current_player,
                "agent_name": acting_agent.name,
                "action": int(action),
                "coords": {"layer": int(layer), "row": int(row), "col": int(col)},
                "elapsed_s": float(elapsed),
                "decision_source": decision_source,
                "decision_label": tactical_reason,
                "search_counted": decision_source == "model_search",
                "board_after": board.tolist(),
            }
            move_history.append(move_record)
            if self.observer is not None:
                self.observer.on_move(game_index, move_record)
            result = game.get_game_ended(board, current_player)
            if result != 0:
                if result == 1e-4:
                    winner_agent = None
                    draw = True
                elif result == -1:
                    winner_agent = player_to_agent[-current_player].name
                    draw = False
                else:
                    winner_agent = player_to_agent[current_player].name
                    draw = False
                result_payload = {
                    "game_index": game_index,
                    "winner": winner_agent,
                    "draw": draw,
                    "illegal_move_by": None,
                    "move_count": move_count,
                    "move_time": total_search_time,
                    "move_decisions": total_search_count,
                    "tactical_moves": tactical_move_count,
                    "first_agent": first_agent.name,
                    "second_agent": second_agent.name,
                    "moves": move_history,
                    "final_board": board.tolist(),
                    "status": "draw" if draw else "finished",
                }
                if self.observer is not None:
                    self.observer.on_game_complete(game_index, result_payload)
                return result_payload

            if self.max_moves is not None and move_count >= int(self.max_moves):
                result = {
                    "game_index": game_index,
                    "winner": None,
                    "draw": True,
                    "illegal_move_by": None,
                    "move_count": move_count,
                    "move_time": total_search_time,
                    "move_decisions": total_search_count,
                    "tactical_moves": tactical_move_count,
                    "first_agent": first_agent.name,
                    "second_agent": second_agent.name,
                    "moves": move_history,
                    "final_board": board.tolist(),
                    "status": "max_moves_draw",
                }
                if self.observer is not None:
                    self.observer.on_game_complete(game_index, result)
                return result

    def _summarize(self, results):
        summary = {
            "games": len(results),
            "agent1": self.agent1.name,
            "agent2": self.agent2.name,
            "agent1_wins": 0,
            "agent2_wins": 0,
            "draws": 0,
            "total_moves": 0,
            "illegal_moves": {
                self.agent1.name: 0,
                self.agent2.name: 0,
            },
            "move_time": {
                self.agent1.name: 0.0,
                self.agent2.name: 0.0,
            },
            "move_decisions": {
                self.agent1.name: 0,
                self.agent2.name: 0,
            },
            "tactical_moves": {
                self.agent1.name: 0,
                self.agent2.name: 0,
            },
            "by_role": {
                self.agent1.name: {"first": 0, "second": 0},
                self.agent2.name: {"first": 0, "second": 0},
            },
            "raw_results": results,
        }

        for result in results:
            summary["total_moves"] += int(result["move_count"])
            summary["by_role"][result["first_agent"]]["first"] += 1
            summary["by_role"][result["second_agent"]]["second"] += 1

            for agent_name in summary["move_time"]:
                summary["move_time"][agent_name] += float(result["move_time"][agent_name])
                summary["move_decisions"][agent_name] += int(result["move_decisions"][agent_name])
                summary["tactical_moves"][agent_name] += int((result.get("tactical_moves") or {}).get(agent_name, 0))

            if result["illegal_move_by"] is not None:
                summary["illegal_moves"][result["illegal_move_by"]] += 1

            if result["draw"]:
                summary["draws"] += 1
            elif result["winner"] == self.agent1.name:
                summary["agent1_wins"] += 1
            elif result["winner"] == self.agent2.name:
                summary["agent2_wins"] += 1

        summary["agent1_win_rate"] = summary["agent1_wins"] / max(1, summary["games"])
        summary["agent2_win_rate"] = summary["agent2_wins"] / max(1, summary["games"])
        summary["draw_rate"] = summary["draws"] / max(1, summary["games"])
        summary["avg_search_time_s"] = {}
        summary["illegal_action_rate"] = {}
        for agent_name in summary["move_time"]:
            decisions = int(summary["move_decisions"][agent_name])
            summary["avg_search_time_s"][agent_name] = summary["move_time"][agent_name] / decisions if decisions > 0 else 0.0
            summary["illegal_action_rate"][agent_name] = summary["illegal_moves"][agent_name] / decisions if decisions > 0 else 0.0
        return summary

    def _select_forced_action(self, game, board, current_player, acting_agent):
        if not self.immediate_win_check or not getattr(acting_agent, "supports_presearch_tactics", False):
            return None

        valid_actions = np.flatnonzero(game.get_valid_moves(board) > 0)
        for action in valid_actions:
            next_board, next_player = game.get_next_state(board, current_player, int(action))
            if game.get_game_ended(next_board, next_player) == -1:
                return {
                    "action": int(action),
                    "source": "forced_win",
                    "label": "Forced Win",
                }

        opponent = -int(current_player)
        for action in valid_actions:
            next_board, next_player = game.get_next_state(board, opponent, int(action))
            if game.get_game_ended(next_board, next_player) == -1:
                return {
                    "action": int(action),
                    "source": "forced_block",
                    "label": "Forced Block",
                }
        return None


def make_game_factory(board_size=5, board_layers=6, connect_n=4):
    def factory():
        return GameRules(board_size=board_size, max_layers=board_layers, connect_n=connect_n)

    return factory


def format_summary(summary):
    lines = [
        f"Total games: {summary['games']}",
        f"{summary['agent1']} wins: {summary['agent1_wins']} | Win rate: {summary['agent1_win_rate']:.3f}",
        f"{summary['agent2']} wins: {summary['agent2_wins']} | Win rate: {summary['agent2_win_rate']:.3f}",
        f"Draws: {summary['draws']} | Draw rate: {summary['draw_rate']:.3f}",
        f"Total moves: {summary['total_moves']}",
    ]
    for agent_name in (summary["agent1"], summary["agent2"]):
        lines.append(
            f"{agent_name} | Avg search time: {summary['avg_search_time_s'][agent_name]:.4f}s | "
            f"Illegal action rate: {summary['illegal_action_rate'][agent_name]:.4f} | "
            f"First-player games: {summary['by_role'][agent_name]['first']} | Second-player games: {summary['by_role'][agent_name]['second']} | "
            f"Tactical direct moves: {summary['tactical_moves'][agent_name]}"
        )
    return "\n".join(lines)
