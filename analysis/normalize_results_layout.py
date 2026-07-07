import argparse
import shutil
from pathlib import Path


RAW_PREFIX = "function_detective__instance_"
GAME_DIR_NAME = "function_detective"


def parse_args():
    parser = argparse.ArgumentParser(description="Flatten Clembench results to results/<mode>/<model>/<experiment>/...")
    parser.add_argument("--results-root", default="results", help="Root results directory to normalize.")
    return parser.parse_args()


def infer_mode_name(raw_dir_name: str) -> str:
    if raw_dir_name.startswith(RAW_PREFIX):
        return raw_dir_name[len(RAW_PREFIX) :]
    return raw_dir_name


def move_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def normalize_mode_dir(mode_root: Path) -> None:
    game_root = mode_root / GAME_DIR_NAME
    if not game_root.exists():
        return

    mode_name = infer_mode_name(mode_root.name)
    final_root = mode_root.parent / mode_name
    final_root.mkdir(parents=True, exist_ok=True)

    for summary_name in ("results.csv", "raw.csv", "results.html"):
        summary_src = mode_root / summary_name
        if summary_src.exists():
            shutil.move(str(summary_src), str(final_root / summary_name))

    for model_dir in sorted(path for path in game_root.iterdir() if path.is_dir()):
        run_json = model_dir / "run.json"
        mode_dir = model_dir / mode_name
        model_target = final_root / model_dir.name
        model_target.mkdir(parents=True, exist_ok=True)

        if run_json.exists():
            shutil.move(str(run_json), str(model_target / "run.json"))

        if mode_dir.exists():
            for experiment_dir in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
                move_tree(experiment_dir, model_target / experiment_dir.name)

    if mode_root.exists():
        shutil.rmtree(mode_root)


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    mode_dirs = [path for path in sorted(results_root.iterdir()) if path.is_dir()]
    for mode_dir in mode_dirs:
        if mode_dir.name.startswith(RAW_PREFIX):
            normalize_mode_dir(mode_dir)


if __name__ == "__main__":
    main()
