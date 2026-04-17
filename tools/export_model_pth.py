from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_SOURCE = Path("save_model") / "v2.2_large" / "best.pth.tar"


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a weights-only model.pth from a checkpoint.")
    parser.add_argument("source", nargs="?", default=str(DEFAULT_SOURCE), help="Input checkpoint path")
    parser.add_argument("output", nargs="?", default=None, help="Output model path")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it exists")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output) if args.output else source.with_name("model.pth")

    if not source.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {source}")
    if output.exists() and not args.force:
        raise FileExistsError(f"Output file already exists: {output}. Use --force to overwrite.")

    checkpoint = load_checkpoint(source)
    state_dict = extract_state_dict(checkpoint)

    torch.save(state_dict, output)
    print(f"Saved weights-only model to: {output}")
    print(f"Source checkpoint: {source}")
    print(f"State dict entries: {len(state_dict) if hasattr(state_dict, '__len__') else 'unknown'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
