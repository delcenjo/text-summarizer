"""Command-line interface: summarise a file or stdin."""

from __future__ import annotations

import argparse
import sys

from .summarizer import key_points, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise text from a file or stdin.")
    parser.add_argument("path", nargs="?", help="Text file to summarise (default: stdin)")
    parser.add_argument("--max-length", type=int, default=130, help="Max summary length")
    parser.add_argument("--min-length", type=int, default=30, help="Min summary length")
    parser.add_argument("--points", action="store_true", help="Print key points instead of a paragraph")
    args = parser.parse_args(argv)

    text = open(args.path, encoding="utf-8").read() if args.path else sys.stdin.read()
    summary = summarize(text, max_length=args.max_length, min_length=args.min_length)

    if args.points:
        for point in key_points(summary):
            print(f"- {point}")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
