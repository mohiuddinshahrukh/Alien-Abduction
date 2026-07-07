import argparse
import shutil
from pathlib import Path

from collect_results import collect_results
from derive_metrics import derive_metrics
from make_figures import make_figures
from make_report import make_report
from make_tables import make_tables


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze one Function Detective results folder.")
    parser.add_argument("--input", required=True, help="Results folder name or absolute path.")
    parser.add_argument("--name", help="Name for the analysis output folder.")
    parser.add_argument("--model", help="Optional exact model name filter within the chosen results folder.")
    return parser.parse_args()


def resolve_input(code_dir: Path, raw_input: str) -> Path:
    candidate = Path(raw_input)
    if candidate.is_absolute():
        return candidate
    return code_dir / raw_input


def main():
    args = parse_args()
    code_dir = Path(__file__).resolve().parents[1]
    input_dir = resolve_input(code_dir, args.input)
    analysis_name = args.name or f"{input_dir.name}_analysis"
    output_dir = code_dir / "analysis_runs" / analysis_name

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = collect_results(input_dir, model_filter=args.model)
    episodes = derive_metrics(episodes)
    episodes.to_csv(output_dir / "episodes.csv", index=False)

    table_paths = make_tables(episodes, output_dir)
    figure_paths = make_figures(episodes, output_dir)
    report_path = make_report(episodes, input_dir, output_dir, figure_paths, table_paths)

    print(f"Analysis written to: {output_dir}")
    print(f"Episodes: {output_dir / 'episodes.csv'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
