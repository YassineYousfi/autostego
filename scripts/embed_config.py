from __future__ import annotations

import argparse
import json
from pathlib import Path

from steganography.embed_dir import EmbedDirConfig, embed_dir


def _load_config(path: Path) -> EmbedDirConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("cover_dir", "stego_dir"):
        if key in raw and raw[key] is not None:
            raw[key] = Path(raw[key])
    return EmbedDirConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a stego directory from a JSON config.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    embed_dir(_load_config(args.config))


if __name__ == "__main__":
    main()
