from __future__ import annotations

import json
import math
import queue
import random
import re
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent
ARENA_DIR = WORKSPACE_ROOT / "arena"
TRAINING_DIR = WORKSPACE_ROOT / "training"

for required_path in (ARENA_DIR, TRAINING_DIR):
    required_path_str = str(required_path)
    if required_path_str not in sys.path:
        sys.path.insert(0, required_path_str)

from agent import MCTSAgent, TinyPolicyAgent, is_tiny_policy_checkpoint
from arena_game_rules import GameRules
from human_eval_config import EVAL_CONFIG, HumanEvalConfig


COLOR_BG = (14, 18, 33)
COLOR_PANEL = (24, 31, 51)
COLOR_TEXT = (228, 232, 243)
COLOR_TEXT_DIM = (136, 148, 174)
COLOR_HUMAN = (235, 99, 73)
COLOR_MODEL = (66, 164, 245)
COLOR_EMPTY = (31, 40, 66)
COLOR_VALID = (62, 213, 120)
COLOR_BORDER = (69, 83, 118)
COLOR_WARN = (245, 186, 61)


def _sanitize_file_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))
    cleaned = cleaned.strip("._")
    return cleaned or "human_eval"


def ensure_history_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_history_file(payload: dict, history_dir: Path, file_stem: str) -> Path:
    target_dir = ensure_history_dir(history_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = _sanitize_file_name(file_stem)
    output_path = target_dir / f"{timestamp}_{safe_stem}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


class HumanModelEvaluator:
    def __init__(self, config: HumanEvalConfig):
        self.config = config
        self.game = GameRules(
            board_size=int(config.board_size),
            max_layers=int(config.board_layers),
            connect_n=int(config.connect_n),
        )
        self.board = self.game.get_init_board()

        self.human_sign = 1 if config.human_plays_first else -1
        self.model_sign = -self.human_sign
        self.current_player = 1

        self.model_agent = None
        self.runtime_info = {}
        self._build_model_agent()

        self.moves = []
        self.model_response_times = []
        self.game_over = False
        self.winner = None
        self.winner_path = [] # Added to track winning line
        self.result_text = ""
        self.saved_history_path = None

        self.ai_running = False
        self.ai_queue = queue.Queue()
        self.click_targets = []

    def _build_model_agent(self) -> None:
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        requested_agent_type = str(getattr(self.config, "agent_type", "auto") or "auto").strip().lower()
        checkpoint_is_tiny = is_tiny_policy_checkpoint(model_path)
        use_tiny = requested_agent_type == "tiny" or (requested_agent_type == "auto" and checkpoint_is_tiny)

        gpu_available = bool(torch.cuda.is_available())
        requested_device = (self.config.device or "").strip().lower() or None
        if requested_device and requested_device.startswith("cuda") and not gpu_available:
            selected_device = "cpu"
            fallback_reason = "requested cuda but no gpu is available"
        elif requested_device:
            selected_device = requested_device
            fallback_reason = None
        else:
            selected_device = "cuda" if gpu_available else "cpu"
            fallback_reason = "gpu unavailable, auto fallback to cpu" if selected_device == "cpu" else None

        if use_tiny:
            linear_cpu_mode = selected_device == "cpu"
            effective_threads = 0
            effective_batch_size = 0
            effective_timeout = 0.0
            self.model_agent = TinyPolicyAgent(
                game=self.game,
                model_path=model_path,
                name=self.config.model_name,
                device=selected_device,
            )
        else:
            linear_cpu_mode = selected_device == "cpu"
            if linear_cpu_mode:
                effective_threads = 1
                effective_batch_size = 1
                effective_timeout = max(0.001, float(self.config.inference_timeout_s))
            else:
                effective_threads = max(1, int(self.config.num_mcts_threads))
                effective_batch_size = max(1, int(self.config.inference_batch_size))
                effective_timeout = float(self.config.inference_timeout_s)

            self.model_agent = MCTSAgent(
                game=self.game,
                model_path=model_path,
                name=self.config.model_name,
                device=selected_device,
                model_config=self.config.model_config,
                num_mcts_sims=int(self.config.num_mcts_sims),
                cpuct=float(self.config.cpuct),
                num_mcts_threads=effective_threads,
                virtual_loss=float(self.config.virtual_loss),
                inference_batch_size=effective_batch_size,
                inference_timeout_s=effective_timeout,
            )

        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        self.runtime_info = {
            "agent_type": "tiny" if use_tiny else "mcts",
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "requested_device": requested_device,
            "effective_device": selected_device,
            "cpu_linear_mode": linear_cpu_mode,
            "fallback_reason": fallback_reason,
            "effective_mcts_threads": effective_threads,
            "effective_inference_batch_size": effective_batch_size,
            "effective_inference_timeout_s": effective_timeout,
        }

    def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Missing pygame. Please install pygame to run human evaluation UI.") from exc

        pygame.init()
        pygame.display.set_caption("Human vs Model Evaluator")
        screen = pygame.display.set_mode((1320, 900), pygame.RESIZABLE)
        clock = pygame.time.Clock()

        title_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        font = pygame.font.SysFont("Segoe UI", 20)
        small_font = pygame.font.SysFont("Consolas", 17)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        continue
                    if event.key == pygame.K_r and self.game_over:
                        self.reset_game()
                        continue
                    if event.key == pygame.K_u:
                        self.undo_move()
                        continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_human_click(event.pos)

            self._consume_ai_result()
            self._start_ai_turn_if_needed()
            self._draw(pygame, screen, title_font, font, small_font)

            pygame.display.flip()
            clock.tick(60)

        self.close()
        pygame.quit()

    def undo_move(self) -> None:
        if self.ai_running or not self.moves:
            return

        # Undo until before the latest human move
        # If the last move was AI, undo AI + Human
        # If game is over, we still want to undo
        
        while self.moves:
            last = self.moves.pop()
            if last["actor"] == "human":
                # We found the human move, now we restore the board to BEFORE this move
                # Actually, the board before human move is the board_after of the move BEFORE human move,
                # or initial board if it was the first move.
                if not self.moves:
                    self.board = self.game.get_init_board()
                else:
                    self.board = np.array(self.moves[-1]["board_after"])
                
                self.current_player = self.human_sign
                self.game_over = False
                self.winner = None
                self.winner_path = []
                self.result_text = ""
                # Clear model times corresponding to the undone moves
                # Since we don't know exactly which model time corresponds to which move in the simple list,
                # we just recalculate from available moves if needed, but the current implementation 
                # just appends. Let's filter model_response_times.
                self.model_response_times = [m["response_s"] for m in self.moves if m["actor"] == "model"]
                break
            else:
                # Undoing an AI move
                continue

    def close(self) -> None:
        if self.model_agent is not None:
            self.model_agent.close()
            self.model_agent = None

    def reset_game(self) -> None:
        self.board = self.game.get_init_board()
        self.current_player = 1
        self.moves = []
        self.model_response_times = []
        self.game_over = False
        self.winner = None
        self.winner_path = []
        self.result_text = ""
        self.saved_history_path = None
        self.ai_running = False
        self.ai_queue = queue.Queue()
        self.click_targets = []

    def _is_human_turn(self) -> bool:
        return (not self.game_over) and self.current_player == self.human_sign and (not self.ai_running)

    def _is_model_turn(self) -> bool:
        return (not self.game_over) and self.current_player == self.model_sign and (not self.ai_running)

    def _start_ai_turn_if_needed(self) -> None:
        if not self._is_model_turn():
            return

        self.ai_running = True
        board_snapshot = np.array(self.board, copy=True)
        player_snapshot = int(self.current_player)

        def worker() -> None:
            try:
                start = time.perf_counter()
                action, _ = self.model_agent.get_action(
                    board_snapshot,
                    player_snapshot,
                    temp=float(self.config.temperature),
                )
                elapsed = time.perf_counter() - start
                self.ai_queue.put(
                    {
                        "ok": True,
                        "action": int(action),
                        "elapsed_s": float(elapsed),
                    }
                )
            except Exception as exc:
                self.ai_queue.put({"ok": False, "error": str(exc)})

        threading.Thread(target=worker, daemon=True).start()

    def _consume_ai_result(self) -> None:
        if not self.ai_running:
            return
        try:
            result = self.ai_queue.get_nowait()
        except queue.Empty:
            return

        self.ai_running = False
        if not result.get("ok"):
            self.game_over = True
            self.result_text = f"Model error: {result.get('error', 'unknown error')}"
            return

        self._apply_action(
            action=int(result["action"]),
            actor="model",
            elapsed_s=float(result["elapsed_s"]),
        )

    def _handle_human_click(self, pos) -> None:
        if not self._is_human_turn():
            return

        for rect, action in self.click_targets:
            if rect.collidepoint(pos):
                self._apply_action(action=int(action), actor="human", elapsed_s=0.0)
                return

    def _apply_action(self, action: int, actor: str, elapsed_s: float) -> None:
        valid_moves = self.game.get_valid_moves(self.board)
        if action < 0 or action >= len(valid_moves) or int(valid_moves[action]) == 0:
            self.game_over = True
            self.winner = "human" if actor == "model" else "model"
            self.result_text = f"Illegal action by {actor}"
            self._save_history()
            return

        self.board, self.current_player = self.game.get_next_state(self.board, self.current_player, int(action))
        acted_player = -int(self.current_player)
        layer, row, col = self.game.action_to_coords(action)

        move_item = {
            "move_index": len(self.moves) + 1,
            "player": int(acted_player),
            "actor": actor,
            "action": int(action),
            "coords": {
                "layer": int(layer),
                "row": int(row),
                "col": int(col),
            },
            "response_s": float(elapsed_s),
            "board_after": self.board.tolist(),
        }
        self.moves.append(move_item)

        if actor == "model":
            self.model_response_times.append(float(elapsed_s))

        self._check_terminal_state()

    def _check_terminal_state(self) -> None:
        ended = self.game.get_game_ended(self.board, self.current_player)
        if ended == 0:
            return

        self.game_over = True
        if ended == 1e-4:
            self.winner = "draw"
            self.result_text = "Result: draw"
        else:
            if ended == -1:
                winner_sign = -int(self.current_player)
            else:
                winner_sign = int(self.current_player)
            self.winner = "human" if winner_sign == self.human_sign else "model"
            self.result_text = f"Result: {self.winner} wins"
            
            # Find the winning line
            self.winner_path = self._find_winning_line(winner_sign)

        self._save_history()

    def _find_winning_line(self, player):
        occupied = np.argwhere(self.board == int(player))
        directions = [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ]

        for layer, row, col in occupied:
            for dz, dy, dx in directions:
                path = [(int(layer), int(row), int(col))]
                for step in range(1, self.game.connect_n):
                    nl, nr, nc = layer + step * dz, row + step * dy, col + step * dx
                    if (0 <= nl < self.game.max_layers and 
                        0 <= nr < self.game.board_size and 
                        0 <= nc < self.game.board_size and 
                        self.board[nl, nr, nc] == player):
                        path.append((int(nl), int(nr), int(nc)))
                    else:
                        break
                if len(path) >= self.game.connect_n:
                    return path
        return []

    def _save_history(self) -> None:
        if self.saved_history_path is not None:
            return

        total_model_time = float(sum(self.model_response_times))
        model_step_count = int(len(self.model_response_times))
        avg_model_time = total_model_time / model_step_count if model_step_count > 0 else 0.0

        config_payload = asdict(self.config)
        config_payload["model_path"] = str(self.config.model_path)
        config_payload["history_dir"] = str(self.config.history_dir)

        payload = {
            "schema_version": 1,
            "title": f"human_vs_{self.config.model_name}",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": config_payload,
            "runtime": dict(self.runtime_info),
            "result": {
                "winner": self.winner,
                "move_count": len(self.moves),
                "result_text": self.result_text,
            },
            "metrics": {
                "model_move_count": model_step_count,
                "model_total_response_s": total_model_time,
                "model_avg_response_s": avg_model_time,
            },
            "moves": list(self.moves),
            "final_board": self.board.tolist(),
        }

        file_stem = f"human_vs_{self.config.model_name}"
        output_path = save_history_file(payload, Path(self.config.history_dir), file_stem=file_stem)
        self.saved_history_path = str(output_path)

    def _draw(self, pygame, screen, title_font, font, small_font) -> None:
        width, height = screen.get_size()
        screen.fill(COLOR_BG)

        board_panel = pygame.Rect(18, 18, width - 380, height - 36)
        side_panel = pygame.Rect(width - 346, 18, 328, height - 36)
        pygame.draw.rect(screen, COLOR_PANEL, board_panel, border_radius=10)
        pygame.draw.rect(screen, COLOR_PANEL, side_panel, border_radius=10)

        self.click_targets = self._draw_board_panel(
            pygame,
            screen,
            board_panel,
            title_font,
            font,
            small_font,
        )
        self._draw_side_panel(pygame, screen, side_panel, font, small_font)

    def _draw_board_panel(self, pygame, screen, panel_rect, title_font, font, small_font):
        clickable = []

        title = "Human vs Model Evaluation"
        subtitle = (
            f"{self.config.human_name} ({'first' if self.config.human_plays_first else 'second'}) "
            f"vs {self.config.model_name}"
        )
        screen.blit(title_font.render(title, True, COLOR_TEXT), (panel_rect.x + 18, panel_rect.y + 16))
        screen.blit(font.render(subtitle, True, COLOR_TEXT_DIM), (panel_rect.x + 18, panel_rect.y + 52))

        status_line = self._status_line()
        status_color = COLOR_WARN if self.ai_running else COLOR_TEXT
        screen.blit(small_font.render(status_line, True, status_color), (panel_rect.x + 18, panel_rect.y + 82))

        board = self.board
        layers = int(board.shape[0])
        board_size = int(board.shape[1])
        columns = min(3, layers)
        rows = max(1, math.ceil(layers / columns))

        layer_gap = 18
        top_offset = 122
        bottom_padding = 18
        title_space_per_layer = 24

        available_w = panel_rect.width - 36 - layer_gap * (columns - 1)
        available_h = panel_rect.height - top_offset - bottom_padding - layer_gap * (rows - 1) - title_space_per_layer * rows
        cell_size = max(
            14,
            min(
                52,
                available_w // max(1, columns * board_size),
                available_h // max(1, rows * board_size),
            ),
        )

        layer_board_px = cell_size * board_size
        valid_moves = self.game.get_valid_moves(board)
        human_turn = self._is_human_turn()
        mouse_pos = pygame.mouse.get_pos()

        # Last AI move marker
        last_ai_move = None
        if self.moves and self.moves[-1]["actor"] == "model":
            last_ai_move = (self.moves[-1]["coords"]["layer"], 
                           self.moves[-1]["coords"]["row"], 
                           self.moves[-1]["coords"]["col"])

        for layer in range(layers):
            grid_col = layer % columns
            grid_row = layer // columns
            origin_x = panel_rect.x + 18 + grid_col * (layer_board_px + layer_gap)
            origin_y = panel_rect.y + top_offset + grid_row * (layer_board_px + layer_gap + title_space_per_layer)

            screen.blit(font.render(f"Layer {layer + 1}", True, COLOR_TEXT), (origin_x, origin_y - 22))

            for row in range(board_size):
                for col in range(board_size):
                    rect = pygame.Rect(
                        origin_x + col * cell_size,
                        origin_y + row * cell_size,
                        cell_size - 2,
                        cell_size - 2,
                    )
                    pygame.draw.rect(screen, COLOR_EMPTY, rect, border_radius=4)
                    pygame.draw.rect(screen, COLOR_BORDER, rect, width=1, border_radius=4)

                    piece = int(board[layer, row, col])
                    if piece != 0:
                        piece_color = COLOR_HUMAN if piece == self.human_sign else COLOR_MODEL
                        center = (rect.x + rect.width // 2, rect.y + rect.height // 2)
                        
                        # Highlight winning path
                        if (layer, row, col) in self.winner_path:
                            # Draw a halo for winning path
                            pygame.draw.circle(screen, (255, 255, 255), center, cell_size // 2, width=2)
                        
                        pygame.draw.circle(screen, piece_color, center, max(4, cell_size // 2 - 5))
                        
                        # AI last move marker
                        if last_ai_move == (layer, row, col):
                            pygame.draw.circle(screen, COLOR_WARN, center, max(12, cell_size // 2), 2)
                        continue

                    action = self.game.coords_to_action(layer, row, col)
                    if human_turn and int(valid_moves[action]) == 1:
                        # Blur-like hover effect
                        if rect.collidepoint(mouse_pos):
                            alpha_color = list(COLOR_HUMAN) + [100]
                            overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                            pygame.draw.circle(overlay, alpha_color, (rect.width//2, rect.height//2), max(4, cell_size // 2 - 5))
                            screen.blit(overlay, rect.topleft)
                        
                        clickable.append((rect, action))

        return clickable

    def _draw_side_panel(self, pygame, screen, panel_rect, font, small_font):
        x = panel_rect.x + 16
        y = panel_rect.y + 16

        lines = [
            f"Model: {self.config.model_name}",
            f"Agent type: {self.runtime_info.get('agent_type')}",
            f"MCTS sims: {self.config.num_mcts_sims}",
            f"Virtual loss: {self.config.virtual_loss}",
            f"Temperature: {self.config.temperature}",
            f"C_puct: {self.config.cpuct}",
            f"Device: {self.runtime_info.get('effective_device')}",
            f"CPU linear mode: {self.runtime_info.get('cpu_linear_mode')}",
            f"MCTS threads: {self.runtime_info.get('effective_mcts_threads')}",
            f"Infer batch: {self.runtime_info.get('effective_inference_batch_size')}",
        ]

        fallback_reason = self.runtime_info.get("fallback_reason")
        if fallback_reason:
            lines.append(f"Fallback: {fallback_reason}")

        lines.extend(
            [
                "",
                f"Moves played: {len(self.moves)}",
                f"Model moves: {len(self.model_response_times)}",
                f"Model avg response: {self._avg_model_time():.4f}s",
                "",
                "Controls:",
                "- Click circles to move",
                "- ESC: exit",
                "- U: undo to your last move",
                "- R: restart after game over",
            ]
        )

        for line in lines:
            if line == "":
                y += 8
                continue
            renderer = font if y < panel_rect.y + 260 else small_font
            screen.blit(renderer.render(line, True, COLOR_TEXT_DIM), (x, y))
            y += 24

        if self.result_text:
            y += 8
            screen.blit(font.render(self.result_text, True, COLOR_TEXT), (x, y))
            y += 28

        if self.saved_history_path:
            screen.blit(small_font.render("History saved:", True, COLOR_TEXT), (x, y))
            y += 22
            for chunk in self._split_text(self.saved_history_path, 38):
                screen.blit(small_font.render(chunk, True, COLOR_VALID), (x, y))
                y += 20

    def _split_text(self, text: str, width: int):
        if len(text) <= width:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + width])
            start += width
        return chunks

    def _avg_model_time(self) -> float:
        if not self.model_response_times:
            return 0.0
        return float(sum(self.model_response_times) / len(self.model_response_times))

    def _status_line(self) -> str:
        if self.game_over:
            if self.saved_history_path:
                return "Game finished. History saved. Press R to start a new game."
            return "Game finished."
        if self.ai_running:
            dot_count = int(time.time() * 2) % 4
            return "Model is thinking" + "." * dot_count
        if self.current_player == self.human_sign:
            return "Your turn: click a highlighted legal move."
        return "Model turn."


def main() -> None:
    config = EVAL_CONFIG

    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    app = HumanModelEvaluator(config)
    app.run()


if __name__ == "__main__":
    main()
