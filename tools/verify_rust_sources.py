#!/usr/bin/env python3
"""Fail if Rust sources contain markdown fence markers (common accident when pasting notes)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "src"
    bad: list[str] = []
    for path in sorted(root.rglob("*.rs")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "```" in text:
            bad.append(str(path))
    if bad:
        print("Markdown fences found in Rust sources (remove ``` lines):", file=sys.stderr)
        for p in bad:
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
