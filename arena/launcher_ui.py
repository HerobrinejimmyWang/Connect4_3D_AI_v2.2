from __future__ import annotations

from pathlib import Path

import pygame

from history import list_replay_sessions
from model_registry import discover_models


COLOR_BG = (11, 16, 38)
COLOR_PANEL = (28, 37, 65)
COLOR_PRIMARY = (0, 168, 232)
COLOR_SECONDARY = (58, 80, 107)
COLOR_TEXT = (224, 224, 224)
COLOR_TEXT_DIM = (120, 130, 150)
COLOR_ACCENT = (0, 255, 153)
COLOR_WARNING = (255, 0, 85)


class ActionButton:
    def __init__(self, rect, text, callback, fill=COLOR_PANEL):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.fill = fill
        self.hovered = False

    def draw(self, screen, font):
        color = tuple(min(255, channel + 22) for channel in self.fill) if self.hovered else self.fill
        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        pygame.draw.rect(screen, COLOR_PRIMARY, self.rect, 1, border_radius=6)
        text_surface = font.render(self.text, True, COLOR_TEXT)
        screen.blit(text_surface, text_surface.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.callback()
            return True
        return False


class LauncherUI:
    def __init__(self, args, workspace_root):
        self.args = args
        self.workspace_root = Path(workspace_root)
        self.history_dir = Path(getattr(args, "history_dir", None) or (self.workspace_root / "arena" / "history"))
        self.model_options = [{"label": "Random Agent", "path": None, "relative_path": "__random__"}] + discover_models(self.workspace_root)
        self.replay_options = list_replay_sessions(self.history_dir, limit=80)
        self.result = None
        self.error_message = None
        self.running = True
        self._screen_size = (1450, 945)
        self._buttons = []
        self._list_rects = {}
        self._scroll = {"p1": 0, "p2": 0, "replay": 0}

        self.state = {
            "mode": "replay" if getattr(args, "history_file", None) else "arena",
            "p1_index": self._resolve_initial_index(getattr(args, "p1_model", None), getattr(args, "p1_random", False)),
            "p2_index": self._resolve_initial_index(getattr(args, "p2_model", None), getattr(args, "p2_random", False)),
            "replay_index": self._resolve_initial_replay_index(getattr(args, "history_file", None)),
            "games": int(getattr(args, "games", 20)),
            "parallel_games": int(getattr(args, "parallel_games", 0)),
            "board_layers": int(getattr(args, "board_layers", 6)),
            "mcts_sims": int(getattr(args, "mcts_sims", 64)),
            "max_moves": int(getattr(args, "max_moves", 0) or 0),
            "temperature": float(getattr(args, "temperature", 0.0)),
            "immediate_win_check": bool(getattr(args, "immediate_win_check", False)),
        }

    def run(self):
        pygame.init()
        pygame.display.set_caption("Arena Configuration")
        screen = pygame.display.set_mode(self._screen_size, pygame.RESIZABLE)
        self._screen_size = screen.get_size()
        clock = pygame.time.Clock()
        title_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
        font = pygame.font.SysFont("Segoe UI", 22)
        small_font = pygame.font.SysFont("Consolas", 18)

        self._rebuild_buttons(screen)
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    self._screen_size = screen.get_size()
                    self._rebuild_buttons(screen)
                    continue
                if self._handle_list_event(event):
                    continue
                for button in self._buttons:
                    if button.handle_event(event):
                        break

            self._draw(screen, title_font, font, small_font)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        return self.result

    def _resolve_initial_index(self, model_path, is_random):
        if is_random:
            return 0
        if model_path:
            target = str(Path(model_path))
            for index, item in enumerate(self.model_options):
                if item["path"] == target:
                    return index
        return 0 if len(self.model_options) == 1 else 1

    def _resolve_initial_replay_index(self, history_file):
        if history_file:
            target = str(Path(history_file))
            for index, item in enumerate(self.replay_options):
                if item["key"] == target:
                    return index
        return 0

    def _rebuild_buttons(self, screen):
        width, height = screen.get_size()
        self._buttons = [
            ActionButton(
                pygame.Rect(42, 70, 140, 38),
                "Arena Mode",
                lambda: self._set_mode("arena"),
                fill=(19, 94, 86) if self.state["mode"] == "arena" else COLOR_SECONDARY,
            ),
            ActionButton(
                pygame.Rect(194, 70, 150, 38),
                "Replay Mode",
                lambda: self._set_mode("replay"),
                fill=(19, 94, 86) if self.state["mode"] == "replay" else COLOR_SECONDARY,
            ),
            ActionButton(pygame.Rect(width - 310, height - 74, 130, 40), "Cancel", self._cancel),
            ActionButton(
                pygame.Rect(width - 164, height - 74, 130, 40),
                "Open Replay" if self.state["mode"] == "replay" else "Start Arena",
                self._confirm,
                fill=(19, 94, 86),
            ),
        ]

        if self.state["mode"] == "arena":
            steppers = [
                ("games", 4, 200, 2),
                ("parallel_games", 0, 32, 1),
                ("board_layers", 4, 8, 1),
                ("mcts_sims", 64, 4096, 64),
                ("max_moves", 0, 300, 5),
            ]
            base_x = 58
            base_y = height - 180
            spacing_x = 230
            for index, (key, min_value, max_value, step) in enumerate(steppers):
                x = base_x + index * spacing_x
                self._buttons.append(
                    ActionButton(
                        pygame.Rect(x, base_y + 42, 36, 32),
                        "-",
                        lambda target=key, lo=min_value, hi=max_value, delta=step: self._change_int(target, -delta, lo, hi),
                    )
                )
                self._buttons.append(
                    ActionButton(
                        pygame.Rect(x + 130, base_y + 42, 36, 32),
                        "+",
                        lambda target=key, lo=min_value, hi=max_value, delta=step: self._change_int(target, delta, lo, hi),
                    )
                )
            temp_x = base_x + 5 * spacing_x
            self._buttons.append(ActionButton(pygame.Rect(temp_x, base_y + 42, 36, 32), "-", lambda: self._change_float("temperature", -0.1, 0.0, 1.0)))
            self._buttons.append(ActionButton(pygame.Rect(temp_x + 130, base_y + 42, 36, 32), "+", lambda: self._change_float("temperature", 0.1, 0.0, 1.0)))
            self._buttons.append(
                ActionButton(
                    pygame.Rect(width - 42 - 250, height - 212, 230, 34),
                    f"Immediate Win Check: {'ON' if self.state['immediate_win_check'] else 'OFF'}",
                    self._toggle_immediate_win_check,
                    fill=(19, 94, 86) if self.state["immediate_win_check"] else COLOR_SECONDARY,
                )
            )

    def _handle_list_event(self, event):
        if self.state["mode"] == "replay":
            return self._handle_replay_list_event(event)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for key, items in self._list_rects.items():
                for index, rect in items:
                    if rect.collidepoint(event.pos):
                        self.state[f"{key}_index"] = index
                        return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
            for key in ("p1", "p2"):
                panel_rect = self._column_panel_rect(key)
                if panel_rect.collidepoint(event.pos):
                    direction = -1 if event.button == 4 else 1
                    self._scroll[key] = max(0, min(max(0, len(self.model_options) - 8), self._scroll[key] + direction))
                    return True
        return False

    def _handle_replay_list_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, rect in self._list_rects.get("replay", []):
                if rect.collidepoint(event.pos):
                    self.state["replay_index"] = index
                    return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
            panel_rect = self._replay_panel_rect()
            if panel_rect.collidepoint(event.pos):
                direction = -1 if event.button == 4 else 1
                self._scroll["replay"] = max(0, min(max(0, len(self.replay_options) - 12), self._scroll["replay"] + direction))
                return True
        return False

    def _draw(self, screen, title_font, font, small_font):
        width, height = screen.get_size()
        screen.fill(COLOR_BG)
        pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(20, 20, width - 40, height - 40), border_radius=12)

        screen.blit(title_font.render("Arena Configuration", True, COLOR_TEXT), (42, 34))
        if self.state["mode"] == "replay":
            subtitle = "Select a completed arena history and open it in the visual replay viewer."
            screen.blit(font.render(subtitle, True, COLOR_TEXT_DIM), (44, 116))
            self._draw_replay_panel(screen, font, small_font)
        else:
            subtitle = "Select models and parameters before starting a new arena session."
            screen.blit(font.render(subtitle, True, COLOR_TEXT_DIM), (44, 116))
            self._draw_column(screen, font, small_font, "p1", "P1 (First) Candidate Models", 42, 164)
            self._draw_column(screen, font, small_font, "p2", "P2 (Second) Candidate Models", width // 2 + 10, 164)
            self._draw_settings(screen, font, small_font)

        if self.error_message:
            screen.blit(font.render(self.error_message, True, COLOR_WARNING), (44, height - 104))

        for button in self._buttons:
            button.draw(screen, font)

    def _draw_column(self, screen, font, small_font, key, title, x, y):
        width, height = screen.get_size()
        panel_width = width // 2 - 58
        panel_height = height - 390
        panel_rect = pygame.Rect(x, y, panel_width, panel_height)
        pygame.draw.rect(screen, tuple(max(0, channel - 8) for channel in COLOR_PANEL), panel_rect, border_radius=10)
        pygame.draw.rect(screen, COLOR_PRIMARY, panel_rect, 1, border_radius=10)
        screen.blit(font.render(title, True, COLOR_TEXT), (x + 16, y + 14))

        selected = self.model_options[self.state[f"{key}_index"]]
        selected_label = selected["relative_path"] if selected["path"] else "Random Agent"
        screen.blit(small_font.render(f"Current: {selected_label}", True, COLOR_ACCENT), (x + 16, y + 42))

        top = y + 76
        visible_items = self.model_options[self._scroll[key] : self._scroll[key] + 8]
        rects = []
        mouse_pos = pygame.mouse.get_pos()
        for row_index, item in enumerate(visible_items):
            actual_index = self._scroll[key] + row_index
            row_rect = pygame.Rect(x + 14, top + row_index * 40, panel_width - 28, 32)
            active = actual_index == self.state[f"{key}_index"]
            is_hovered = row_rect.collidepoint(mouse_pos)
            
            if active:
                fill = tuple(min(255, c + 30) for c in COLOR_PRIMARY) if is_hovered else COLOR_PRIMARY
            else:
                fill = tuple(min(255, c + 30) for c in COLOR_SECONDARY) if is_hovered else COLOR_SECONDARY
            
            pygame.draw.rect(screen, fill, row_rect, border_radius=6)
            
            display_label = item["label"]
            if len(display_label) > 60:
                display_label = display_label[:57] + "..."
            
            screen.blit(small_font.render(display_label, True, COLOR_TEXT), (row_rect.x + 10, row_rect.y + 7))
            rects.append((actual_index, row_rect))
        self._list_rects[key] = rects

        if len(self.model_options) > 8:
            hint = f"Scroll to flip {self._scroll[key] + 1}-{min(len(self.model_options), self._scroll[key] + 8)} / {len(self.model_options)}"
            screen.blit(small_font.render(hint, True, COLOR_TEXT_DIM), (x + 16, panel_rect.bottom - 28))

    def _draw_replay_panel(self, screen, font, small_font):
        panel_rect = self._replay_panel_rect()
        pygame.draw.rect(screen, tuple(max(0, channel - 8) for channel in COLOR_PANEL), panel_rect, border_radius=10)
        pygame.draw.rect(screen, COLOR_PRIMARY, panel_rect, 1, border_radius=10)
        screen.blit(font.render("Completed Arena Histories", True, COLOR_TEXT), (panel_rect.x + 16, panel_rect.y + 14))
        screen.blit(small_font.render(f"History folder: {self.history_dir}", True, COLOR_TEXT_DIM), (panel_rect.x + 16, panel_rect.y + 44))

        if not self.replay_options:
            self._list_rects["replay"] = []
            screen.blit(font.render("No completed histories found.", True, COLOR_WARNING), (panel_rect.x + 16, panel_rect.y + 92))
            screen.blit(small_font.render("Run at least one arena session first so it is saved into history.", True, COLOR_TEXT_DIM), (panel_rect.x + 16, panel_rect.y + 126))
            return

        visible_items = self.replay_options[self._scroll["replay"] : self._scroll["replay"] + 12]
        rects = []
        mouse_pos = pygame.mouse.get_pos()
        top = panel_rect.y + 82
        selected_index = max(0, min(self.state["replay_index"], len(self.replay_options) - 1))
        self.state["replay_index"] = selected_index

        for row_index, item in enumerate(visible_items):
            actual_index = self._scroll["replay"] + row_index
            row_rect = pygame.Rect(panel_rect.x + 14, top + row_index * 54, panel_rect.width - 28, 46)
            active = actual_index == selected_index
            is_hovered = row_rect.collidepoint(mouse_pos)
            if active:
                fill = tuple(min(255, c + 30) for c in COLOR_PRIMARY) if is_hovered else COLOR_PRIMARY
            else:
                fill = tuple(min(255, c + 30) for c in COLOR_SECONDARY) if is_hovered else COLOR_SECONDARY

            pygame.draw.rect(screen, fill, row_rect, border_radius=6)
            summary = item.get("summary") or {}
            title = item["label"]
            if len(title) > 72:
                title = title[:69] + "..."
            detail = f"{item['file_name']} | Games {item['games']} | W {summary.get('agent1_wins', 0)}:{summary.get('agent2_wins', 0)} | D {summary.get('draws', 0)}"
            screen.blit(small_font.render(title, True, COLOR_TEXT), (row_rect.x + 10, row_rect.y + 6))
            screen.blit(small_font.render(detail[:110], True, COLOR_TEXT_DIM), (row_rect.x + 10, row_rect.y + 24))
            rects.append((actual_index, row_rect))
        self._list_rects["replay"] = rects

        selected = self.replay_options[selected_index]
        summary = selected.get("summary") or {}
        info_y = panel_rect.bottom - 84
        info_line = (
            f"Selected: {selected['file_name']} | Saved: {selected.get('saved_at') or 'unknown'} | "
            f"Total {summary.get('games', selected['games'])} games"
        )
        screen.blit(small_font.render(info_line[:140], True, COLOR_ACCENT), (panel_rect.x + 16, info_y))
        if len(self.replay_options) > 12:
            hint = f"Scroll to flip {self._scroll['replay'] + 1}-{min(len(self.replay_options), self._scroll['replay'] + 12)} / {len(self.replay_options)}"
            screen.blit(small_font.render(hint, True, COLOR_TEXT_DIM), (panel_rect.x + 16, panel_rect.bottom - 42))

    def _draw_settings(self, screen, font, small_font):
        width, height = screen.get_size()
        panel_rect = pygame.Rect(42, height - 222, width - 84, 116)
        pygame.draw.rect(screen, tuple(max(0, channel - 8) for channel in COLOR_PANEL), panel_rect, border_radius=10)
        pygame.draw.rect(screen, COLOR_PRIMARY, panel_rect, 1, border_radius=10)
        screen.blit(font.render("Basic Settings", True, COLOR_TEXT), (panel_rect.x + 16, panel_rect.y + 12))

        items = [
            ("Games", str(self.state["games"]), panel_rect.x + 16),
            ("Parallel", "AUTO" if self.state["parallel_games"] == 0 else str(self.state["parallel_games"]), panel_rect.x + 246),
            ("Layers", str(self.state["board_layers"]), panel_rect.x + 476),
            ("MCTS Sims", str(self.state["mcts_sims"]), panel_rect.x + 706),
            ("Max Moves", "NONE" if self.state["max_moves"] == 0 else str(self.state["max_moves"]), panel_rect.x + 936),
            ("Temp", f"{self.state['temperature']:.1f}", panel_rect.x + 1166),
        ]
        for label, value, x in items:
            screen.blit(small_font.render(label, True, COLOR_TEXT_DIM), (x, panel_rect.y + 42))
            screen.blit(font.render(value, True, COLOR_TEXT), (x + 50, panel_rect.y + 74))

    def _column_panel_rect(self, key):
        width, height = self._screen_size
        x = 42 if key == "p1" else width // 2 + 10
        y = 164
        panel_width = width // 2 - 58
        panel_height = height - 390
        return pygame.Rect(x, y, panel_width, panel_height)

    def _replay_panel_rect(self):
        width, height = self._screen_size
        return pygame.Rect(42, 164, width - 84, height - 254)

    def _change_int(self, key, delta, min_value, max_value):
        value = int(self.state[key]) + int(delta)
        self.state[key] = max(int(min_value), min(int(max_value), value))

    def _change_float(self, key, delta, min_value, max_value):
        value = round(float(self.state[key]) + float(delta), 1)
        self.state[key] = max(float(min_value), min(float(max_value), value))

    def _cancel(self):
        self.result = None
        self.running = False

    def _toggle_immediate_win_check(self):
        self.state["immediate_win_check"] = not self.state["immediate_win_check"]
        surface = pygame.display.get_surface()
        if surface is not None:
            self._rebuild_buttons(surface)

    def _set_mode(self, mode):
        if mode not in {"arena", "replay"}:
            return
        self.state["mode"] = mode
        self.error_message = None
        surface = pygame.display.get_surface()
        if surface is not None:
            self._rebuild_buttons(surface)

    def _confirm(self):
        if self.state["mode"] == "replay":
            if not self.replay_options:
                self.error_message = "No replayable history records were found."
                return
            replay_index = max(0, min(self.state["replay_index"], len(self.replay_options) - 1))
            selected = self.replay_options[replay_index]
            self.result = {
                "history_file": str(selected["path"]),
                "history_dir": str(self.history_dir),
            }
            self.running = False
            return

        p1 = self.model_options[self.state["p1_index"]]
        p2 = self.model_options[self.state["p2_index"]]
        self.result = {
            "p1_model": p1["path"],
            "p1_random": p1["path"] is None,
            "p1_name": _default_agent_name(p1),
            "p1_mcts_sims": int(self.state["mcts_sims"]),
            "p2_model": p2["path"],
            "p2_random": p2["path"] is None,
            "p2_name": _default_agent_name(p2),
            "p2_mcts_sims": int(self.state["mcts_sims"]),
            "games": int(self.state["games"]),
            "parallel_games": int(self.state["parallel_games"]),
            "board_layers": int(self.state["board_layers"]),
            "mcts_sims": int(self.state["mcts_sims"]),
            "max_moves": None if int(self.state["max_moves"]) == 0 else int(self.state["max_moves"]),
            "temperature": float(self.state["temperature"]),
            "immediate_win_check": bool(self.state["immediate_win_check"]),
        }
        self.running = False


def _default_agent_name(item):
    if item["path"] is None:
        return "random"
    return Path(item["path"]).parent.name