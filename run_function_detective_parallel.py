#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    instance_name: str
    results_dir: str


RUN_SPECS = [
    RunSpec("instance_active_inputs", "results_active_inputs"),
    RunSpec("instance_active_pair_checks", "results_active_pair_checks"),
    RunSpec("instance_passive_examples", "results_passive_examples"),
    RunSpec("instance_passive_examples_oneshot", "results_passive_examples_oneshot"),
    RunSpec("instance_passive_labeled_pairs", "results_passive_labeled_pairs"),
    RunSpec("instance_passive_labeled_pairs_oneshot", "results_passive_labeled_pairs_oneshot"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Function Detective instance sets in parallel for one model."
    )
    parser.add_argument("model", help="Model name passed to `clem run -m`.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=len(RUN_SPECS),
        help="How many runs to execute concurrently. Default: all.",
    )
    parser.add_argument(
        "--game",
        default="function_detective",
        help="Game name passed to `clem run -g`. Default: function_detective.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def build_command(game: str, model: str, spec: RunSpec) -> list[str]:
    return [
        "clem",
        "run",
        "-g",
        game,
        "-m",
        model,
        "-i",
        spec.instance_name,
        "-r",
        spec.results_dir,
    ]


def run_one(game: str, model: str, spec: RunSpec, logs_dir: Path) -> tuple[RunSpec, int, Path]:
    cmd = build_command(game, model, spec)
    log_path = logs_dir / f"{model}__{spec.instance_name}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        process = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    return spec, process.returncode, log_path


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    logs_dir = repo_root / "logs" / "parallel_runs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    commands = [(spec, build_command(args.game, args.model, spec)) for spec in RUN_SPECS]

    if args.dry_run:
        for _, cmd in commands:
            print(" ".join(cmd))
        return 0

    max_workers = max(1, min(args.max_workers, len(RUN_SPECS)))
    print(f"Running {len(RUN_SPECS)} jobs for model={args.model} with max_workers={max_workers}")
    print(f"Logs: {logs_dir}")

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, args.game, args.model, spec, logs_dir): spec
            for spec in RUN_SPECS
        }

        for future in as_completed(futures):
            spec, returncode, log_path = future.result()
            status = "OK" if returncode == 0 else f"FAIL ({returncode})"
            print(f"[{status}] {spec.instance_name} -> {spec.results_dir} | log={log_path}")
            if returncode != 0:
                failures.append((spec, returncode, log_path))

    if failures:
        print("\nFailed runs:")
        for spec, returncode, log_path in failures:
            print(f"- {spec.instance_name}: exit={returncode} log={log_path}")
        return 1

    print("\nAll runs finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
