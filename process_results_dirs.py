#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


GAME_NAME = "function_detective"


def find_result_dirs(repo_root: Path) -> list[Path]:
    return sorted(path for path in repo_root.iterdir() if path.is_dir() and path.name.startswith("results"))


def run_command(cmd: list[str], cwd: Path) -> int:
    print(f"$ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    return completed.returncode


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    result_dirs = find_result_dirs(repo_root)

    if not result_dirs:
        print("No directories starting with 'results' found.")
        return 1

    failures: list[tuple[str, str, int]] = []

    for result_dir in result_dirs:
        print(f"\n==> Processing {result_dir.name}")

        commands = [
            ["clem", "transcribe", "-g", GAME_NAME, "-r", result_dir.name],
            ["clem", "score", "-g", GAME_NAME, "-r", result_dir.name],
            ["clem", "eval", "-r", result_dir.name],
        ]

        for cmd in commands:
            returncode = run_command(cmd, cwd=repo_root)
            if returncode != 0:
                failures.append((result_dir.name, " ".join(cmd), returncode))
                print(f"Command failed with exit code {returncode}. Continuing to next step/folder.")

    if failures:
        print("\nFailures:")
        for folder, cmd, returncode in failures:
            print(f"- {folder}: exit={returncode} | {cmd}")
        return 1

    print("\nAll result directories processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
