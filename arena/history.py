from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_DIR = CURRENT_DIR / "history"


def _sanitize_file_name(text):
    sanitized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(text))
    sanitized = sanitized.strip("._")
    return sanitized or "arena_session"


def ensure_history_dir(path=None):
    history_dir = Path(path) if path else DEFAULT_HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def save_session(session_data, history_dir=None, file_stem=None):
    target_dir = ensure_history_dir(history_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _sanitize_file_name(file_stem or session_data.get("title") or "arena_session")
    file_path = target_dir / f"{timestamp}_{stem}.json"
    payload = deepcopy(session_data)
    payload.setdefault("schema_version", 1)
    if _is_replayable_session(payload):
        payload["completed"] = True
    else:
        payload.setdefault("completed", False)
    payload.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def load_session(file_path):
    path = Path(file_path)
    return json.loads(path.read_text(encoding="utf-8"))


def list_history_files(history_dir=None, limit=30):
    target_dir = ensure_history_dir(history_dir)
    files = sorted(target_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[: max(1, int(limit))]


def list_replay_sessions(history_dir=None, limit=30):
    sessions = []
    max_count = max(1, int(limit))
    for file_path in list_history_files(history_dir, limit=max_count * 3):
        try:
            session = load_session(file_path)
        except (OSError, json.JSONDecodeError):
            continue

        if not _is_replayable_session(session):
            continue

        summary = session.get("summary") or {}
        sessions.append(
            {
                "path": file_path,
                "key": str(file_path),
                "label": session.get("title") or file_path.stem,
                "file_name": file_path.name,
                "saved_at": session.get("saved_at"),
                "summary": summary,
                "games": len(session.get("games") or []),
            }
        )
        if len(sessions) >= max_count:
            break
    return sessions


def _is_replayable_session(session):
    games = session.get("games") or []
    if not games:
        return False

    if session.get("completed") is True:
        return True

    if session.get("summary") and session.get("error") in (None, ""):
        return True

    terminal_statuses = {"finished", "draw", "illegal_move", "max_moves_draw"}
    return all((game.get("status") in terminal_statuses) for game in games)


class ArenaSessionRecorder:
    def __init__(self, session_meta):
        self._lock = threading.Lock()
        self._games = {}
        self._session_meta = deepcopy(session_meta)
        self._summary = None
        self._saved_path = None
        self._error = None
        self._completed = False

    def on_game_start(self, game_index, first_agent, second_agent, initial_board):
        with self._lock:
            self._games[int(game_index)] = {
                "game_index": int(game_index),
                "status": "running",
                "first_agent": first_agent,
                "second_agent": second_agent,
                "winner": None,
                "draw": False,
                "illegal_move_by": None,
                "move_count": 0,
                "moves": [],
                "initial_board": _to_serializable_board(initial_board),
                "final_board": _to_serializable_board(initial_board),
            }

    def on_move(self, game_index, move_record):
        with self._lock:
            game = self._games[int(game_index)]
            game["moves"].append(deepcopy(move_record))
            game["move_count"] = len(game["moves"])
            game["final_board"] = deepcopy(move_record["board_after"])

    def on_game_complete(self, game_index, result):
        with self._lock:
            game = self._games.setdefault(
                int(game_index),
                {
                    "game_index": int(game_index),
                    "moves": [],
                    "initial_board": [],
                    "final_board": [],
                },
            )
            for key, value in deepcopy(result).items():
                game[key] = value
            if game.get("illegal_move_by"):
                game["status"] = "illegal_move"
            elif game.get("draw"):
                game["status"] = "draw"
            else:
                game["status"] = "finished"

    def finalize(self, summary=None, saved_path=None, error=None):
        with self._lock:
            self._summary = deepcopy(summary) if summary is not None else None
            self._saved_path = str(saved_path) if saved_path else None
            self._error = str(error) if error else None
            self._completed = True

    def snapshot(self):
        with self._lock:
            games = [deepcopy(self._games[key]) for key in sorted(self._games)]
            payload = deepcopy(self._session_meta)
            payload["games"] = games
            payload["summary"] = deepcopy(self._summary)
            payload["saved_path"] = self._saved_path
            payload["error"] = self._error
            payload["completed"] = self._completed
            payload.setdefault("schema_version", 1)
            return payload


def _to_serializable_board(board):
    if hasattr(board, "tolist"):
        return board.tolist()
    return deepcopy(board)