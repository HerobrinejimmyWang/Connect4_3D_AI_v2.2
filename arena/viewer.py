from __future__ import annotations

import math
from pathlib import Path

from history import list_history_files, load_session


COLOR_BG = (11, 16, 38)
COLOR_PANEL = (28, 37, 65)
COLOR_PRIMARY = (0, 168, 232)
COLOR_SECONDARY = (58, 80, 107)
COLOR_TEXT = (224, 224, 224)
COLOR_TEXT_DIM = (120, 130, 150)
COLOR_ACCENT = (0, 255, 153)
COLOR_WARNING = (255, 0, 85)
COLOR_RED = (220, 70, 70)
COLOR_BLUE = (60, 150, 220)
COLOR_EMPTY = (19, 25, 44)


class Button:
    def __init__(self, rect, text, callback):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.hovered = False

    def draw(self, pygame, screen, font):
        color = tuple(min(255, channel + 40) for channel in COLOR_PANEL) if self.hovered else COLOR_PANEL 
        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        pygame.draw.rect(screen, COLOR_PRIMARY, self.rect, 1, border_radius=6)
        text_surface = font.render(self.text, True, COLOR_TEXT)
        screen.blit(text_surface, text_surface.get_rect(center=self.rect.center))

    def handle_event(self, pygame, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.callback()
            return True
        return False


class ArenaViewer:
    def __init__(self, live_recorder=None, history_dir=None):
        self.live_recorder = live_recorder
        self.history_dir = Path(history_dir) if history_dir else None
        self.selected_game_index = 0
        self.selected_move_index = 0
        self.selected_source_key = "live"
        self.selected_history_path = None
        self.autoplay = True
        self.autoplay_speed_ms = 450
        self.last_autoplay_tick = 0
        self.current_snapshot = None
        self._cached_history = {}
        self._games_scroll = 0
        self.return_to_launcher = False
        self._buttons = []
        self._screen_size = (1500, 940)

    def run(self):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Missing pygame, cannot open arena viewer.") from exc

        pygame.init()
        pygame.display.set_caption("Arena Viewer")
        screen = pygame.display.set_mode(self._screen_size, pygame.RESIZABLE)
        self._screen_size = screen.get_size()
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Segoe UI", 20)
        title_font = pygame.font.SysFont("Segoe UI", 26, bold=True)
        small_font = pygame.font.SysFont("Consolas", 18)

        self._build_buttons(pygame, screen)

        self.running = True
        while self.running:
            now = pygame.time.get_ticks()
            snapshot = self._resolve_selected_snapshot()
            if snapshot is not None:
                self.current_snapshot = snapshot

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    continue
                if event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    self._screen_size = screen.get_size()
                    self._build_buttons(pygame, screen)
                    continue
                if self._handle_input(pygame, event):
                    continue
                for button in self._buttons:
                    if button.handle_event(pygame, event):
                        break

            if self.autoplay and self.current_snapshot and now - self.last_autoplay_tick >= self.autoplay_speed_ms:
                self._advance_move()
                self.last_autoplay_tick = now

            self._draw(pygame, screen, title_font, font, small_font)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def _resolve_selected_snapshot(self):
        if self.selected_source_key == "live":
            return self.live_recorder.snapshot() if self.live_recorder is not None else None
        history_path = Path(self.selected_source_key)
        cache_key = str(history_path)
        if cache_key not in self._cached_history:
            self._cached_history[cache_key] = load_session(history_path)
        return self._cached_history[cache_key]

    def _build_buttons(self, pygame, screen):
        width, height = screen.get_size()
        button_y = height - 58
        self._buttons = [
            Button(pygame.Rect(24, button_y, 110, 34), "Prev Move", self._rewind_move),
            Button(pygame.Rect(144, button_y, 110, 34), "Next Move", self._advance_move),
            Button(pygame.Rect(264, button_y, 130, 34), "Autoplay", self._toggle_autoplay),
            Button(pygame.Rect(404, button_y, 150, 34), "Play from Start", self._play_from_start),
            Button(pygame.Rect(564, button_y, 110, 34), "Prev Game", self._previous_game),
            Button(pygame.Rect(684, button_y, 110, 34), "Next Game", self._next_game),
            Button(pygame.Rect(804, button_y, 150, 34), "Back to Menu", self._go_back_to_menu),
        ]

    def _handle_input(self, pygame, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self._rewind_move()
                return True
            if event.key == pygame.K_RIGHT:
                self._advance_move()
                return True
            if event.key == pygame.K_UP:
                self._previous_game()
                return True
            if event.key == pygame.K_DOWN:
                self._next_game()
                return True
            if event.key == pygame.K_SPACE:
                self._toggle_autoplay()
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
            width, height = self._screen_size
            if event.pos[0] >= width - 320:
                direction = -1 if event.button == 4 else 1
                num_games = len(self.current_snapshot.get("games", [])) if self.current_snapshot else 0
                max_scroll = max(0, num_games - 18)
                self._games_scroll = max(0, min(max_scroll, self._games_scroll + direction))
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.current_snapshot is not None:
            return self._handle_list_click(event.pos)
        return False

    def _handle_list_click(self, pos):
        if self.current_snapshot is None:
            return False

        width, height = self._screen_size
        x, y = pos
        num_games = len(self.current_snapshot.get("games", []))
        if x >= width - 320 and 48 <= y <= 48 + 18 * 34:
            index = (y - 48) // 34 + self._games_scroll
            if 0 <= index < num_games:
                self.selected_game_index = int(index)
                game = self.current_snapshot.get("games", [])[self.selected_game_index]
                self.selected_move_index = len(game.get("moves", []))
                self.autoplay = False
                return True
        return False

    def _selected_game(self):
        if self.current_snapshot is None:
            return None
        games = self.current_snapshot.get("games", [])
        if not games:
            return None
        self.selected_game_index = max(0, min(self.selected_game_index, len(games) - 1))
        return games[self.selected_game_index]

    def _rewind_move(self):
        self.autoplay = False
        self.selected_move_index = max(0, self.selected_move_index - 1)

    def _advance_move(self):
        game = self._selected_game()
        if game is None:
            return
        total_moves = len(game.get("moves", []))
        if total_moves == 0:
            self.selected_move_index = 0
            return
        if self.selected_move_index < total_moves:
            self.selected_move_index += 1

    def _toggle_autoplay(self):
        self.autoplay = not self.autoplay

    def _play_from_start(self):
        self.selected_move_index = 0
        self.autoplay = True

    def _previous_game(self):
        self.autoplay = False
        self.selected_game_index = max(0, self.selected_game_index - 1)
        game = self._selected_game()
        self.selected_move_index = len(game.get("moves", [])) if game else 0

    def _next_game(self):
        if self.current_snapshot is None:
            return
        self.autoplay = False
        game_count = len(self.current_snapshot.get("games", []))
        if game_count == 0:
            return
        self.selected_game_index = min(game_count - 1, self.selected_game_index + 1)
        game = self._selected_game()
        self.selected_move_index = len(game.get("moves", [])) if game else 0

    def _go_back_to_menu(self):
        self.return_to_launcher = True
        self.running = False

    def _draw(self, pygame, screen, title_font, font, small_font):
        width, height = screen.get_size()
        screen.fill(COLOR_BG)
        snapshot = self.current_snapshot

        pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(16, 16, width - 352, height - 32), border_radius=10)
        pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(width - 320, 16, 304, height - 32), border_radius=10)

        if snapshot is None:
            text = title_font.render("No arena records available", True, COLOR_TEXT)
            screen.blit(text, (40, 40))
            return

        title = snapshot.get("title") or "Arena Viewer"
        screen.blit(title_font.render(title, True, COLOR_TEXT), (30, 26))

        summary = snapshot.get("summary") or {}
        subtitle = self._summary_line(summary)
        screen.blit(font.render(subtitle, True, COLOR_TEXT_DIM), (30, 60))
        if snapshot.get("saved_path"):
            screen.blit(small_font.render(f"Saved: {snapshot['saved_path']}", True, COLOR_ACCENT), (30, 88))
        elif snapshot.get("completed"):
            screen.blit(small_font.render("Completed, waiting to save results", True, COLOR_ACCENT), (30, 88))
        else:
            screen.blit(small_font.render("In Progress", True, COLOR_WARNING), (30, 88))

        game = self._selected_game()
        self._draw_board_panel(pygame, screen, font, small_font, game, width, height)
        self._draw_sidebar(pygame, screen, font, small_font, snapshot, width, height)

        for button in self._buttons:
            button.draw(pygame, screen, font)

    def _draw_board_panel(self, pygame, screen, font, small_font, game, width, height):
        panel_rect = pygame.Rect(28, 146, width - 380, height - 232)
        pygame.draw.rect(screen, tuple(max(0, channel - 8) for channel in COLOR_PANEL), panel_rect, border_radius=8)
        if game is None:
            screen.blit(font.render("Waiting for game data...", True, COLOR_TEXT_DIM), (44, 140))
            return

        board = self._board_for_display(game)
        board_size = len(board[0]) if board else 0
        layers = len(board)

        # Better logic for finding number of columns
        # Each board width is around 150-200. Let's start by trying to fit as many as possible
        tile_gap = 18
        max_possible_cols = max(1, (panel_rect.width - 36 + tile_gap) // (120 + tile_gap))
        columns = min(layers, max_possible_cols) if layers > 0 else 1
        # Try to balance rows and columns
        if columns > 1:
            while columns > 1 and math.ceil(layers / columns) == math.ceil(layers / (columns - 1)):
                columns -= 1
        
        rows = max(1, math.ceil(layers / columns))
        available_width = panel_rect.width - 36 - tile_gap * (columns - 1)
        available_height = panel_rect.height - 70 - tile_gap * (rows - 1)
        
        cell_size = max(12, min(34, available_width // max(1, columns * max(1, board_size)), available_height // max(1, rows * max(1, board_size))))
        layer_board_size = cell_size * board_size
        start_x = panel_rect.x + 18
        start_y = panel_rect.y + 18

        curr_x = start_x
        for text, color in self._game_status_text_blocks(game):
            surf = font.render(text, True, color)
            screen.blit(surf, (curr_x, start_y))
            curr_x += surf.get_width()

        curr_x = start_x
        for text, color in self._move_status_text_blocks(game):
            surf = small_font.render(text, True, color)
            screen.blit(surf, (curr_x, start_y + 28))
            curr_x += surf.get_width()

        # Check if we need to swap colors
        # p1 should always be red, p2 always blue.
        # In engine, '1' is the first_agent. If P2 is first_agent, '1' should be rendered as blue.
        swap_colors = False
        snapshot = self.current_snapshot
        if snapshot:
            agents = snapshot.get("agents", {})
            if agents:
                p2_name = agents.get("agent2", {}).get("name")
            else:
                p2_name = snapshot.get("summary", {}).get("agent2")
            if game.get("first_agent") == p2_name:
                swap_colors = True

        for layer_index in range(layers):
            grid_col = layer_index % columns
            grid_row = layer_index // columns
            origin_x = start_x + grid_col * (layer_board_size + tile_gap)
            origin_y = start_y + 66 + grid_row * (layer_board_size + tile_gap + 28)
            self._draw_single_layer(pygame, screen, font, board[layer_index], layer_index, origin_x, origin_y, cell_size, swap_colors)

    def _draw_single_layer(self, pygame, screen, font, layer_board, layer_index, origin_x, origin_y, cell_size, swap_colors):
        size = len(layer_board)
        screen.blit(font.render(f"Layer {layer_index + 1}", True, COLOR_TEXT), (origin_x, origin_y - 24))
        for row in range(size):
            for col in range(size):
                rect = pygame.Rect(origin_x + col * cell_size, origin_y + row * cell_size, cell_size - 2, cell_size - 2)
                pygame.draw.rect(screen, COLOR_SECONDARY, rect, border_radius=3)
                piece = int(layer_board[row][col])
                color = COLOR_EMPTY
                if piece > 0:
                    color = COLOR_BLUE if swap_colors else COLOR_RED
                elif piece < 0:
                    color = COLOR_RED if swap_colors else COLOR_BLUE
                center = (rect.x + rect.width // 2, rect.y + rect.height // 2)
                pygame.draw.circle(screen, color, center, max(3, cell_size // 2 - 4))

    def _draw_sidebar(self, pygame, screen, font, small_font, snapshot, width, height):
        x = width - 304
        mouse_pos = pygame.mouse.get_pos()

        screen.blit(font.render("Games", True, COLOR_TEXT), (x, 22))
        
        games = snapshot.get("games", [])
        visible_games = games[self._games_scroll : self._games_scroll + 18]

        for row_index, game in enumerate(visible_games):
            actual_index = self._games_scroll + row_index
            rect = pygame.Rect(x, 48 + row_index * 34, 272, 28)
            active = actual_index == self.selected_game_index
            is_hovered = rect.collidepoint(mouse_pos)
            
            if active:
                fill = tuple(min(255, c + 40) for c in COLOR_PRIMARY) if is_hovered else COLOR_PRIMARY
            else:
                fill = tuple(min(255, c + 40) for c in COLOR_SECONDARY) if is_hovered else COLOR_SECONDARY
                
            pygame.draw.rect(screen, fill, rect, border_radius=4)
            indicator_color = self._game_result_color(game)
            pygame.draw.circle(screen, indicator_color, (rect.x + 12, rect.y + rect.height // 2), 5)
            label = self._game_list_label(game)
            screen.blit(small_font.render(label[:30], True, COLOR_TEXT), (rect.x + 24, rect.y + 6))

        if len(games) > 18:
            hint = f"Scroll: {self._games_scroll + 1}-{min(len(games), self._games_scroll + 18)} / {len(games)}"
            screen.blit(small_font.render(hint, True, COLOR_TEXT_DIM), (x, 48 + 18 * 34 + 6))

        screen.blit(font.render("Instructions", True, COLOR_TEXT), (x, height - 170))
        hints = [
            "Left/Right: Prev/Next Move",
            "Up/Down: Switch Game",
            "Space: Toggle Autoplay",
        ]
        for index, line in enumerate(hints):
            screen.blit(small_font.render(line, True, COLOR_TEXT_DIM), (x, height - 144 + index * 22))

    def _get_agent_names(self):
        snapshot = self.current_snapshot or {}
        agents = snapshot.get("agents", {})
        if agents:
            return agents.get("agent1", {}).get("name", "P1"), agents.get("agent2", {}).get("name", "P2")
        summary = snapshot.get("summary", {})
        return summary.get("agent1", "P1"), summary.get("agent2", "P2")

    def _summary_line(self, summary):
        if not summary:
            return "Waiting for summary..."
        return (
            f"Games: {summary.get('games', 0)}  |  "
            f"RED Win: {summary.get('agent1_wins', 0)}  |  "
            f"BLUE Win: {summary.get('agent2_wins', 0)}  |  "
            f"Draw: {summary.get('draws', 0)}"
        )

    def _game_status_text_blocks(self, game):
        p1_name, p2_name = self._get_agent_names()
        blocks = []
        blocks.append((f"Game {game.get('game_index', 0) + 1} | First: ", COLOR_TEXT))
        
        first_agent = game.get("first_agent")
        if first_agent == p2_name:
            blocks.append((str(first_agent), COLOR_BLUE))
            blocks.append((" | Second: ", COLOR_TEXT))
            blocks.append((str(game.get("second_agent") or p1_name), COLOR_RED))
        else:
            blocks.append((str(first_agent or p1_name), COLOR_RED))
            blocks.append((" | Second: ", COLOR_TEXT))
            blocks.append((str(game.get("second_agent") or p2_name), COLOR_BLUE))
            
        blocks.append((" | ", COLOR_TEXT))
        
        winner = game.get("winner")
        if game.get("illegal_move_by"):
            blocks.append((f"Illegal move by: {game['illegal_move_by']}", COLOR_WARNING))
        elif game.get("draw"):
            blocks.append(("Result: DRAW", COLOR_TEXT))
        elif winner:
            if winner == p1_name:
                blocks.append(("RED WIN", COLOR_RED))
            elif winner == p2_name:
                blocks.append(("BLUE WIN", COLOR_BLUE))
            else:
                blocks.append((f"WIN {winner}", COLOR_TEXT))
        else:
            blocks.append((f"Status: {game.get('status', 'running')}", COLOR_TEXT))
            
        return blocks

    def _move_status_text_blocks(self, game):
        p1_name, p2_name = self._get_agent_names()
        total_moves = len(game.get("moves", []))
        shown_index = min(self.selected_move_index, total_moves)
        if shown_index == 0:
            return [(f"Replay: 0 / {total_moves}", COLOR_TEXT_DIM)]
        
        move = game["moves"][shown_index - 1]
        coords = move.get("coords", {})
        agent_name = move.get('agent_name')
        
        c = COLOR_BLUE if agent_name == p2_name else COLOR_RED
        
        blocks = [
            (f"Replay: {shown_index} / {total_moves} | Move {move.get('move_number')} | ", COLOR_TEXT_DIM),
            (str(agent_name), c),
            (f" -> L{coords.get('layer', 0) + 1} R{coords.get('row', 0) + 1} C{coords.get('col', 0) + 1}", COLOR_TEXT_DIM)
        ]
        
        source_label = self._decision_source_label(move)
        if source_label:
            blocks.append((source_label, COLOR_TEXT_DIM))
            
        return blocks

    def _game_list_label(self, game):
        p1_name, p2_name = self._get_agent_names()
        if game.get("illegal_move_by"):
            suffix = "Illegal"
        elif game.get("draw"):
            suffix = "Draw"
        elif game.get("winner"):
            winner = game.get("winner")
            if winner == p1_name:
                suffix = "Red Win"
            elif winner == p2_name:
                suffix = "Blue Win"
            else:
                suffix = f"Win {winner}"
        else:
            suffix = game.get("status", "Running")
        return f"#{game.get('game_index', 0) + 1:02d}  {suffix}  {game.get('move_count', 0)} moves"

    def _game_result_color(self, game):
        p1_name, p2_name = self._get_agent_names()
        if game.get("illegal_move_by"):
            return COLOR_WARNING
        winner = game.get("winner")
        if winner == p1_name:
            return COLOR_RED
        if winner == p2_name:
            return COLOR_BLUE
        return COLOR_TEXT_DIM

    def _decision_source_label(self, move):
        label = move.get("decision_label")
        if label:
            return f" | {label}"
        source = move.get("decision_source")
        if source == "forced_win":
            return " | Forced Win"
        if source == "forced_block":
            return " | Forced Block"
        return ""

    def _board_for_display(self, game):
        moves = game.get("moves", [])
        if self.selected_move_index <= 0:
            return game.get("initial_board") or []
        if self.selected_move_index <= len(moves):
            return moves[self.selected_move_index - 1].get("board_after") or []
        return game.get("final_board") or []