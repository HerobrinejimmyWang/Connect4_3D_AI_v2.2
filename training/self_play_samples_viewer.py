#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


COLOR_BG = (16, 20, 32)
COLOR_PANEL = (30, 36, 56)
COLOR_CARD = (40, 48, 72)
COLOR_CARD_ACTIVE = (0, 155, 220)
COLOR_GRID = (70, 82, 115)
COLOR_TEXT = (228, 232, 242)
COLOR_TEXT_DIM = (152, 162, 188)
COLOR_ACCENT = (0, 230, 160)
COLOR_WARNING = (255, 170, 60)
COLOR_P1 = (220, 70, 70)
COLOR_P2 = (60, 150, 220)
COLOR_EMPTY = (22, 28, 45)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


class SelfPlaySamplesViewer:
    def __init__(self, json_path: Path, layers: int = 6, rows: int = 5, cols: int = 5):
        self.json_path = json_path
        self.layers = max(1, int(layers))
        self.rows = max(1, int(rows))
        self.cols = max(1, int(cols))

        self.payload = self._load_json(json_path)
        self.samples = list(self.payload.get("samples", []))

        self.selected_sample_index = 0
        self.selected_move_index = 0
        self.autoplay = False
        self.autoplay_speed_ms = 350
        self.last_autoplay_tick = 0

        self.running = True
        self.screen_size = (1500, 940)

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object.")
        if "samples" not in data or not isinstance(data.get("samples"), list):
            raise ValueError("Invalid self_play_samples.json: missing samples list.")
        return data

    def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

        pygame.init()
        pygame.display.set_caption("Self-Play Samples Viewer")
        screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        self.screen_size = screen.get_size()

        font = pygame.font.SysFont("Consolas", 18)
        title_font = pygame.font.SysFont("Segoe UI", 26, bold=True)
        small_font = pygame.font.SysFont("Consolas", 16)

        clock = pygame.time.Clock()

        while self.running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    continue
                if event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    self.screen_size = screen.get_size()
                    continue
                self._handle_event(pygame, event)

            if self.autoplay and now - self.last_autoplay_tick >= self.autoplay_speed_ms:
                self._next_move(loop_sample=True)
                self.last_autoplay_tick = now

            self._draw(pygame, screen, title_font, font, small_font)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def _handle_event(self, pygame, event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.autoplay = False
                self._prev_move()
            elif event.key == pygame.K_RIGHT:
                self.autoplay = False
                self._next_move(loop_sample=False)
            elif event.key == pygame.K_UP:
                self.autoplay = False
                self._prev_sample()
            elif event.key == pygame.K_DOWN:
                self.autoplay = False
                self._next_sample()
            elif event.key == pygame.K_SPACE:
                self.autoplay = not self.autoplay
            elif event.key == pygame.K_HOME:
                self.selected_move_index = 0
            elif event.key == pygame.K_END:
                self.selected_move_index = len(self._selected_moves())

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_mouse_click(event.pos)

    def _handle_mouse_click(self, pos: tuple[int, int]) -> None:
        x, y = pos
        width, _ = self.screen_size
        sidebar_x = width - 350

        if x < sidebar_x:
            return

        if 84 <= y <= 84 + 16 * 34:
            local = (y - 84) // 34
            idx = _clamp(int(local), 0, max(0, len(self.samples) - 1))
            self.selected_sample_index = idx
            self.selected_move_index = len(self._selected_moves())

    def _selected_sample(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        self.selected_sample_index = _clamp(self.selected_sample_index, 0, len(self.samples) - 1)
        sample = self.samples[self.selected_sample_index]
        if not isinstance(sample, dict):
            return None
        return sample

    def _selected_moves(self) -> list[dict[str, Any]]:
        sample = self._selected_sample()
        if sample is None:
            return []
        moves = sample.get("moves", [])
        if not isinstance(moves, list):
            return []
        result = []
        for item in moves:
            if isinstance(item, dict):
                result.append(item)
        return result

    def _prev_move(self) -> None:
        self.selected_move_index = max(0, self.selected_move_index - 1)

    def _next_move(self, loop_sample: bool) -> None:
        moves = self._selected_moves()
        if self.selected_move_index < len(moves):
            self.selected_move_index += 1
            return
        if loop_sample:
            self.selected_move_index = 0

    def _prev_sample(self) -> None:
        if not self.samples:
            return
        self.selected_sample_index = max(0, self.selected_sample_index - 1)
        self.selected_move_index = len(self._selected_moves())

    def _next_sample(self) -> None:
        if not self.samples:
            return
        self.selected_sample_index = min(len(self.samples) - 1, self.selected_sample_index + 1)
        self.selected_move_index = len(self._selected_moves())

    def _decode_action(self, action: int) -> tuple[int, int, int]:
        volume = self.rows * self.cols
        layer = action // volume
        rem = action % volume
        row = rem // self.cols
        col = rem % self.cols
        return layer, row, col

    def _reconstruct_board_at_move(self, move_count: int) -> list[list[list[int]]]:
        board = [
            [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            for _ in range(self.layers)
        ]
        moves = self._selected_moves()
        use_count = _clamp(move_count, 0, len(moves))

        for i in range(use_count):
            move = moves[i]
            action = _safe_int(move.get("action"), -1)
            player = _safe_int(move.get("player"), 0)
            layer, row, col = self._decode_action(action)
            if 0 <= layer < self.layers and 0 <= row < self.rows and 0 <= col < self.cols:
                board[layer][row][col] = 1 if player >= 0 else -1
        return board

    def _draw(self, pygame, screen, title_font, font, small_font) -> None:
        width, height = self.screen_size
        screen.fill(COLOR_BG)

        left_panel = pygame.Rect(16, 16, width - 382, height - 32)
        right_panel = pygame.Rect(width - 350, 16, 334, height - 32)

        pygame.draw.rect(screen, COLOR_PANEL, left_panel, border_radius=10)
        pygame.draw.rect(screen, COLOR_PANEL, right_panel, border_radius=10)

        self._draw_header(screen, title_font, font)
        self._draw_board_area(pygame, screen, font, small_font, left_panel)
        self._draw_sidebar(pygame, screen, font, small_font, right_panel)
        self._draw_footer(screen, small_font, height)

    def _draw_header(self, screen, title_font, font) -> None:
        title = "Self-Play JSON Viewer"
        subtitle = f"File: {self.json_path}"
        screen.blit(title_font.render(title, True, COLOR_TEXT), (30, 24))
        screen.blit(font.render(subtitle, True, COLOR_TEXT_DIM), (30, 58))

    def _draw_board_area(self, pygame, screen, font, small_font, panel_rect) -> None:
        sample = self._selected_sample()
        if sample is None:
            screen.blit(font.render("No samples in this JSON.", True, COLOR_WARNING), (panel_rect.x + 16, panel_rect.y + 18))
            return

        moves = self._selected_moves()
        self.selected_move_index = _clamp(self.selected_move_index, 0, len(moves))

        info_line = (
            f"Sample {self.selected_sample_index + 1}/{len(self.samples)}"
            f" | step {self.selected_move_index}/{len(moves)}"
            f" | steps={_safe_int(sample.get('steps'), 0)}"
            f" | entropy={float(sample.get('policy_entropy_mean', 0.0)):.4f}"
        )
        screen.blit(font.render(info_line, True, COLOR_TEXT), (panel_rect.x + 16, panel_rect.y + 16))

        outcome = self._sample_outcome_text(sample)
        screen.blit(small_font.render(outcome, True, COLOR_ACCENT), (panel_rect.x + 16, panel_rect.y + 44))

        board = self._reconstruct_board_at_move(self.selected_move_index)
        self._draw_layers_grid(pygame, screen, font, panel_rect, board)

        last_move_line = self._current_move_text()
        screen.blit(small_font.render(last_move_line, True, COLOR_TEXT_DIM), (panel_rect.x + 16, panel_rect.bottom - 28))

    def _draw_layers_grid(self, pygame, screen, font, panel_rect, board) -> None:
        total_layers = len(board)
        if total_layers <= 0:
            return

        top = panel_rect.y + 78
        bottom = panel_rect.bottom - 52
        usable_h = max(100, bottom - top)
        usable_w = panel_rect.width - 24

        gap = 16
        cols = max(1, min(total_layers, int((usable_w + gap) // (120 + gap))))
        rows = max(1, math.ceil(total_layers / cols))

        layer_w = (usable_w - gap * (cols - 1)) // cols
        layer_h = (usable_h - gap * (rows - 1)) // rows
        square = min(layer_w, layer_h)

        cell_size = max(10, min(34, square // max(self.rows, self.cols)))
        board_w = self.cols * cell_size
        board_h = self.rows * cell_size

        origin_x = panel_rect.x + 12
        origin_y = top

        for i in range(total_layers):
            grid_col = i % cols
            grid_row = i // cols
            x = origin_x + grid_col * (square + gap)
            y = origin_y + grid_row * (square + gap)
            self._draw_one_layer(pygame, screen, font, board[i], i, x, y, board_w, board_h, cell_size)

    def _draw_one_layer(self, pygame, screen, font, layer_board, layer_idx, x, y, board_w, board_h, cell_size) -> None:
        label = f"L{layer_idx + 1}"
        screen.blit(font.render(label, True, COLOR_TEXT), (x, y - 24))

        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(x + c * cell_size, y + r * cell_size, cell_size - 2, cell_size - 2)
                pygame.draw.rect(screen, COLOR_GRID, rect, border_radius=3)
                piece = int(layer_board[r][c])
                color = COLOR_EMPTY
                if piece > 0:
                    color = COLOR_P1
                elif piece < 0:
                    color = COLOR_P2
                center = (rect.x + rect.width // 2, rect.y + rect.height // 2)
                pygame.draw.circle(screen, color, center, max(3, cell_size // 2 - 3))

        outer = pygame.Rect(x - 2, y - 2, board_w + 4, board_h + 4)
        pygame.draw.rect(screen, COLOR_CARD, outer, width=1, border_radius=4)

    def _draw_sidebar(self, pygame, screen, font, small_font, panel_rect) -> None:
        screen.blit(font.render("Samples", True, COLOR_TEXT), (panel_rect.x + 12, panel_rect.y + 12))

        max_rows = 16
        for i in range(min(max_rows, len(self.samples))):
            sample = self.samples[i]
            y = panel_rect.y + 68 + i * 34
            rect = pygame.Rect(panel_rect.x + 10, y, panel_rect.width - 20, 28)
            active = i == self.selected_sample_index
            color = COLOR_CARD_ACTIVE if active else COLOR_CARD
            pygame.draw.rect(screen, color, rect, border_radius=5)

            label = self._sample_label(i, sample)
            screen.blit(small_font.render(label[:46], True, COLOR_TEXT), (rect.x + 10, rect.y + 6))

        sample = self._selected_sample()
        if sample is not None:
            y0 = panel_rect.y + 68 + max_rows * 34 + 8
            screen.blit(font.render("Selected", True, COLOR_TEXT), (panel_rect.x + 12, y0))

            detail_lines = self._sample_details(sample)
            for idx, line in enumerate(detail_lines):
                color = COLOR_TEXT_DIM
                if "winner" in line.lower() or "draw" in line.lower():
                    color = COLOR_ACCENT
                screen.blit(small_font.render(line, True, color), (panel_rect.x + 12, y0 + 30 + idx * 22))

    def _draw_footer(self, screen, small_font, height: int) -> None:
        controls = [
            "Left/Right: prev/next move",
            "Up/Down: prev/next sample",
            "Space: autoplay",
            "Home/End: begin/end",
        ]
        x = 24
        for line in controls:
            surf = small_font.render(line, True, COLOR_TEXT_DIM)
            screen.blit(surf, (x, height - 28))
            x += surf.get_width() + 26

    def _sample_label(self, idx: int, sample: dict[str, Any]) -> str:
        steps = _safe_int(sample.get("steps"), 0)
        winner = _safe_int(sample.get("winner"), 0)
        is_draw = bool(sample.get("is_draw", False))
        if is_draw:
            outcome = "Draw"
        elif winner > 0:
            outcome = "P1 Win"
        elif winner < 0:
            outcome = "P2 Win"
        else:
            outcome = "Unknown"
        return f"#{idx + 1:02d}  {outcome}  {steps} moves"

    def _sample_details(self, sample: dict[str, Any]) -> list[str]:
        details = []
        details.append(f"sample_rank: {_safe_int(sample.get('sample_rank'), 0)}")
        details.append(f"game_idx: {_safe_int(sample.get('game_idx'), -1)}")
        details.append(f"steps: {_safe_int(sample.get('steps'), 0)}")
        details.append(f"used_for_training: {bool(sample.get('used_for_training', False))}")
        details.append(f"policy_entropy_mean: {float(sample.get('policy_entropy_mean', 0.0)):.4f}")

        winner = _safe_int(sample.get("winner"), 0)
        is_draw = bool(sample.get("is_draw", False))
        if is_draw:
            details.append("result: draw")
        elif winner > 0:
            details.append("result: winner P1")
        elif winner < 0:
            details.append("result: winner P2")
        else:
            details.append("result: unknown")

        details.append(f"result_code: {float(sample.get('result_code', 0.0)):.4f}")
        return details

    def _sample_outcome_text(self, sample: dict[str, Any]) -> str:
        winner = _safe_int(sample.get("winner"), 0)
        is_draw = bool(sample.get("is_draw", False))
        if is_draw:
            return "Outcome: Draw"
        if winner > 0:
            return "Outcome: Winner is P1 (red)"
        if winner < 0:
            return "Outcome: Winner is P2 (blue)"
        return "Outcome: Unknown"

    def _current_move_text(self) -> str:
        moves = self._selected_moves()
        if self.selected_move_index <= 0 or not moves:
            return "Move: start position"
        if self.selected_move_index > len(moves):
            return "Move: terminal position"

        move = moves[self.selected_move_index - 1]
        step = _safe_int(move.get("step"), self.selected_move_index)
        player = _safe_int(move.get("player"), 0)
        action = _safe_int(move.get("action"), -1)
        layer, row, col = self._decode_action(action)
        player_name = "P1" if player >= 0 else "P2"
        return (
            f"Move {step}: {player_name} action={action} -> "
            f"L{layer + 1} R{row + 1} C{col + 1}"
        )


def _latest_json_in_dir(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in checkpoint_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint_"):
            continue
        suffix = child.name.split("_", 1)[-1]
        try:
            index = int(suffix)
        except ValueError:
            continue
        sample = child / "self_play_samples.json"
        if sample.exists():
            candidates.append((index, sample))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone pygame viewer for self_play_samples.json")
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to self_play_samples.json (if omitted, auto-pick latest from training/checkpoints)",
    )
    parser.add_argument("--layers", type=int, default=6, help="Board layers, default 6")
    parser.add_argument("--rows", type=int, default=5, help="Board rows, default 5")
    parser.add_argument("--cols", type=int, default=5, help="Board cols, default 5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.json_path is not None:
        json_path = args.json_path
    else:
        root = Path(__file__).resolve().parent
        default_dir = root / "checkpoints"
        json_path = _latest_json_in_dir(default_dir)
        if json_path is None:
            raise FileNotFoundError(
                "No self_play_samples.json found under training/checkpoints/checkpoint_*/"
            )

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    viewer = SelfPlaySamplesViewer(
        json_path=json_path,
        layers=args.layers,
        rows=args.rows,
        cols=args.cols,
    )
    viewer.run()


if __name__ == "__main__":
    main()
