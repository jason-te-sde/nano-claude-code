"""The ``ncc`` command."""

from __future__ import annotations

import argparse

from nanoclaude import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ncc", description="A terminal coding agent that brings its own models."
    )
    parser.add_argument("--version", action="version", version=f"nano-claude-code {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        # argparse's own ``--version``/``--help`` handling, and its own argument
        # errors, all terminate by calling ``parser.exit()``, which raises
        # SystemExit rather than returning. Convert that back into a return code
        # so this function never exits the process itself — the console-script
        # entry point (``ncc = "nanoclaude.cli.main:main"``) does that once, at
        # the top, with our return value.
        return exc.code if isinstance(exc.code, int) else 1
    parser.print_usage()
    print("error: nothing to do — pass a prompt, or run: ncc init")
    return 2
