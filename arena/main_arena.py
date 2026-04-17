from __future__ import annotations

import argparse
import multiprocessing
import random
import threading
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent


def add_agent_args(parser, prefix):
    parser.add_argument(f"--{prefix}-model", type=str, help=f"{prefix} model path")
    parser.add_argument(f"--{prefix}-name", type=str, help=f"{prefix} display name")
    parser.add_argument(f"--{prefix}-device", type=str, help=f"{prefix} inference device, e.g. cpu / cuda")
    parser.add_argument(
        f"--{prefix}-agent-type",
        type=str,
        choices=["auto", "mcts", "tiny"],
        default="auto",
        help=f"{prefix} agent type: auto detect, mcts, or tiny",
    )
    parser.add_argument(f"--{prefix}-random", action="store_true", help=f"{prefix} use a random agent")
    parser.add_argument(f"--{prefix}-layers", type=int, help=f"{prefix} model board layers")
    parser.add_argument(f"--{prefix}-board-size", type=int, help=f"{prefix} model board size")
    parser.add_argument(f"--{prefix}-channels", type=int, help=f"{prefix} backbone channels")
    parser.add_argument(f"--{prefix}-res-blocks", type=int, help=f"{prefix} residual block count")
    parser.add_argument(f"--{prefix}-policy-channels", type=int, help=f"{prefix} policy head channels")
    parser.add_argument(f"--{prefix}-value-channels", type=int, help=f"{prefix} value head channels")
    parser.add_argument(f"--{prefix}-value-hidden-dim", type=int, help=f"{prefix} value head hidden width")
    parser.add_argument(f"--{prefix}-dropout", type=float, help=f"{prefix} dropout")
    parser.add_argument(f"--{prefix}-mcts-sims", type=int, default=128, help=f"{prefix} MCTS simulations")
    parser.add_argument(f"--{prefix}-cpuct", type=float, default=1.0, help=f"{prefix} PUCT coefficient")
    parser.add_argument(f"--{prefix}-threads", type=int, default=8, help=f"{prefix} MCTS threads")
    parser.add_argument(f"--{prefix}-virtual-loss", type=float, default=1.0, help=f"{prefix} virtual loss")
    parser.add_argument(f"--{prefix}-inference-batch-size", type=int, default=32, help=f"{prefix} inference batch size")
    parser.add_argument(f"--{prefix}-inference-timeout", type=float, default=0.003, help=f"{prefix} inference batch timeout in seconds")


def build_model_config(args, prefix):
    return {
        "board_layers": getattr(args, f"{prefix}_layers"),
        "board_size": getattr(args, f"{prefix}_board_size"),
        "num_channels": getattr(args, f"{prefix}_channels"),
        "num_res_blocks": getattr(args, f"{prefix}_res_blocks"),
        "policy_channels": getattr(args, f"{prefix}_policy_channels"),
        "value_channels": getattr(args, f"{prefix}_value_channels"),
        "value_hidden_dim": getattr(args, f"{prefix}_value_hidden_dim"),
        "dropout": getattr(args, f"{prefix}_dropout"),
    }


def build_agent(args, prefix, game):
    from agent import MCTSAgent, RandomAgent, TinyPolicyAgent, is_tiny_policy_checkpoint

    if getattr(args, f"{prefix}_random"):
        return RandomAgent(game, getattr(args, f"{prefix}_name") or f"{prefix}_random")

    model_path = getattr(args, f"{prefix}_model")
    if not model_path:
        raise ValueError(f"--{prefix}-model is required unless --{prefix}-random is set")

    requested_type = (getattr(args, f"{prefix}_agent_type") or "auto").strip().lower()
    auto_is_tiny = is_tiny_policy_checkpoint(model_path)
    use_tiny = requested_type == "tiny" or (requested_type == "auto" and auto_is_tiny)

    if use_tiny:
        return TinyPolicyAgent(
            game=game,
            model_path=model_path,
            name=getattr(args, f"{prefix}_name"),
            device=getattr(args, f"{prefix}_device"),
        )

    return MCTSAgent(
        game=game,
        model_path=model_path,
        name=getattr(args, f"{prefix}_name"),
        device=getattr(args, f"{prefix}_device"),
        model_config=build_model_config(args, prefix),
        num_mcts_sims=getattr(args, f"{prefix}_mcts_sims"),
        cpuct=getattr(args, f"{prefix}_cpuct"),
        num_mcts_threads=getattr(args, f"{prefix}_threads"),
        virtual_loss=getattr(args, f"{prefix}_virtual_loss"),
        inference_batch_size=getattr(args, f"{prefix}_inference_batch_size"),
        inference_timeout_s=getattr(args, f"{prefix}_inference_timeout"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="3D Connect Four model arena")
    parser.add_argument("--games", type=int, default=20, help="total number of games")
    parser.add_argument("--parallel-games", type=int, default=0, help="number of parallel games, 0 means auto")
    parser.add_argument("--temperature", type=float, default=0.0, help="move sampling temperature, 0 means greedy")
    parser.add_argument("--board-size", type=int, default=5, help="board size")
    parser.add_argument("--board-layers", type=int, default=6, help="board layers, default 6")
    parser.add_argument("--connect-n", type=int, default=4, help="target connection length")
    parser.add_argument("--max-moves", type=int, help="maximum moves per game before forcing a draw")
    parser.add_argument("--immediate-win-check", action="store_true", help="use tactical immediate win/block checks before model search")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--history-file", type=str, help="open a saved history file instead of starting a new arena")
    parser.add_argument("--history-dir", type=str, help="history directory, defaults to arena/history")
    parser.add_argument("--no-ui", action="store_true", help="run without the visualization window")
    add_agent_args(parser, "p1")
    add_agent_args(parser, "p2")
    return parser.parse_args()


def main():
    multiprocessing.freeze_support()
    original_args = parse_args()

    while True:
        import copy
        args = copy.copy(original_args)

        if not args.no_ui and not args.history_file:
            from launcher_ui import LauncherUI

            launch_config = LauncherUI(args, WORKSPACE_ROOT).run()
            if launch_config is None:
                return
            for key, value in launch_config.items():
                setattr(args, key, value)

        import numpy as np
        import torch

        from arena import Arena, format_summary, make_game_factory
        from history import ArenaSessionRecorder, load_session, save_session
        from viewer import ArenaViewer

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        if getattr(args, "history_file", None):
            if args.no_ui:
                session = load_session(args.history_file)
                print(format_summary(session.get("summary") or {}))
                return
            viewer = ArenaViewer(live_recorder=None, history_dir=args.history_dir or str(Path(args.history_file).resolve().parent))
            viewer.selected_source_key = str(args.history_file)
            viewer.run()
            if viewer.return_to_launcher:
                original_args.history_file = None  # Clear history file to return to main UI
                continue
            return

        if args.parallel_games <= 0:
            cpu_workers = max(1, multiprocessing.cpu_count() // 2)
            args.parallel_games = min(args.games, cpu_workers if torch.cuda.is_available() else max(1, multiprocessing.cpu_count() // 3))

        game_factory = make_game_factory(
            board_size=args.board_size,
            board_layers=args.board_layers,
            connect_n=args.connect_n,
        )
        agent_game = game_factory()

        agent1 = build_agent(args, "p1", agent_game)
        agent2 = build_agent(args, "p2", agent_game)

        session_title = f"{agent1.name}_vs_{agent2.name}"
        recorder = ArenaSessionRecorder(
            {
                "title": session_title,
                "config": {
                    "games": int(args.games),
                    "parallel_games": int(args.parallel_games),
                    "temperature": float(args.temperature),
                    "board_size": int(args.board_size),
                    "board_layers": int(args.board_layers),
                    "connect_n": int(args.connect_n),
                    "max_moves": args.max_moves,
                    "immediate_win_check": bool(args.immediate_win_check),
                    "seed": int(args.seed),
                },
                "agents": {
                    "agent1": {
                        "name": agent1.name,
                        "agent_type": getattr(agent1, "agent_type", "unknown"),
                        "model_path": str(getattr(agent1, "model_path", "random")),
                        "model_config": getattr(agent1, "model_config", None),
                    },
                    "agent2": {
                        "name": agent2.name,
                        "agent_type": getattr(agent2, "agent_type", "unknown"),
                        "model_path": str(getattr(agent2, "model_path", "random")),
                        "model_config": getattr(agent2, "model_config", None),
                    },
                },
            }
        )

        arena = Arena(
            agent1=agent1,
            agent2=agent2,
            game_factory=game_factory,
            num_games=args.games,
            parallel_games=args.parallel_games,
            temperature=args.temperature,
            max_moves=args.max_moves,
            immediate_win_check=args.immediate_win_check,
            observer=recorder,
        )

        result_holder = {"summary": None, "error": None}

        def run_arena_and_save():
            try:
                result_holder["summary"] = arena.play()
                save_path = save_session(
                    {
                        **recorder.snapshot(),
                        "summary": result_holder["summary"],
                    },
                    history_dir=args.history_dir,
                    file_stem=session_title,
                )
                recorder.finalize(summary=result_holder["summary"], saved_path=save_path)
            except Exception as exc:
                result_holder["error"] = exc
                recorder.finalize(error=exc)
            finally:
                arena.close()

        if args.no_ui:
            run_arena_and_save()
        else:
            worker = threading.Thread(target=run_arena_and_save, daemon=True)
            worker.start()
            viewer = ArenaViewer(live_recorder=recorder, history_dir=args.history_dir)
            viewer.run()
            worker.join()
            if viewer.return_to_launcher:
                continue

        if result_holder["error"] is not None:
            raise result_holder["error"]

        print(format_summary(result_holder["summary"]))
        break


if __name__ == "__main__":
    main()
