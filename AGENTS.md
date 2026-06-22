# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python 3D Connect4 AI training and evaluation stack.
Core self-play, MCTS, model, and trainer code lives in `training/`. Match and replay tooling lives in `arena/`, including `main_arena.py`, UI launch code, model discovery, and history handling. Distillation workflows live in `distillation/`; compact feature-based policy training lives in `train_features/`; human evaluation scripts and saved play histories live in `test/`. Utility scripts are in `tools/`. Model checkpoints, logs, caches, and exported weights are stored under paths such as `training/checkpoints/`, `distillation/checkpoints/`, `save_model/`, and `other_models/`.

## Build, Test, and Development Commands

Use the local virtual environment when available:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run lightweight syntax validation before committing:

```powershell
python -m compileall training arena distillation train_features test tools
```

Start major workflows from the repository root:

```powershell
python training\main_train.py
python distillation\main_distill.py --config distillation\distill_config.json --print-config
python arena\main_arena.py --black-random --white-random --games 1
python test\main_human_eval.py
python tools\export_model_pth.py save_model\v2.2_large\best.pth.tar
```

Many scripts are GPU/CPU intensive; reduce iteration counts or use dry configuration flags when checking changes.

## Coding Style & Naming Conventions

Follow existing Python style: 4-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase constants such as `BOARD_SIZE`. Keep script entry points guarded with `if __name__ == "__main__":`, especially where multiprocessing is used. Prefer `pathlib.Path` for new path handling and keep imports explicit. Preserve existing local-import patterns unless converting a full package boundary.

## Testing Guidelines

There is no formal pytest suite configured. Treat `compileall` as the minimum check, then run the smallest relevant workflow: random arena games for game-rule or agent changes, `--print-config` for distillation config changes, and targeted human-history training commands for `train_features/`. Avoid committing generated `__pycache__/`, large experimental checkpoints, or ad hoc logs unless they are intentional artifacts.

## Commit & Pull Request Guidelines

Git history currently contains only `Initial code-only import`, so use concise imperative commits going forward, for example `Add tiny policy cache validation`. Pull requests should describe the affected workflow, list commands run, note hardware assumptions such as CUDA availability, and include screenshots or saved history paths for UI/evaluation changes.

## Agent-Specific Instructions

Keep changes scoped. Do not rewrite checkpoint directories or historical JSON results unless asked. When adding new generated outputs, document where they are produced and whether they should be tracked.
