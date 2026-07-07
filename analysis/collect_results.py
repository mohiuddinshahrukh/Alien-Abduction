import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REQUIRED_TOP_LEVEL_FILES = ("results.csv", "raw.csv")
REQUIRED_EPISODE_FILES = ("scores.json", "instance.json", "interactions.json")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def _normalize_domain(category: str) -> str:
    return str(category or "").strip().lower()


def validate_results_folder(input_dir: Path) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Results folder does not exist: {input_dir}")
    for filename in REQUIRED_TOP_LEVEL_FILES:
        _require_path(input_dir / filename, f"top-level file `{filename}`")
    if not any(input_dir.rglob("scores.json")):
        raise FileNotFoundError(f"No episode `scores.json` files found under: {input_dir}")


def collect_results(input_dir: Path, model_filter: Optional[str] = None) -> pd.DataFrame:
    validate_results_folder(input_dir)

    rows: List[Dict[str, Any]] = []
    for scores_path in sorted(input_dir.rglob("scores.json")):
        episode_dir = scores_path.parent
        instance_path = episode_dir / "instance.json"
        interactions_path = episode_dir / "interactions.json"
        experiment_path = episode_dir.parent / "experiment.json"
        relative_parts = scores_path.relative_to(input_dir).parts

        for filename in REQUIRED_EPISODE_FILES:
            _require_path(episode_dir / filename, f"episode file `{filename}`")
        _require_path(experiment_path, "experiment file `experiment.json`")

        scores = _load_json(scores_path)
        instance = _load_json(instance_path)
        interactions = _load_json(interactions_path)
        experiment = _load_json(experiment_path)

        meta = scores.get("meta", {})
        episode_scores = scores.get("episode scores", {})
        players = scores.get("players", {})
        player_model = (
            players.get("Player 1", {}).get("model_name")
            or (relative_parts[0] if len(relative_parts) >= 4 else scores_path.parts[-5])
        )

        row = (
            {
                "results_dir": input_dir.name,
                "model": player_model,
                "mode": instance.get("mode") or experiment.get("mode") or interactions.get("mode"),
                "domain": _normalize_domain(instance.get("category")),
                "experiment": meta.get("experiment_name") or experiment.get("name"),
                "callable": instance.get("callable"),
                "game_id": instance.get("game_id", meta.get("game_id")),
                "played": True,
                "round_count": meta.get("round_count", interactions.get("meta", {}).get("round_count", 0)),
                "success": bool(interactions.get("Success", episode_scores.get("Success", False))),
                "lose": bool(interactions.get("Lose", episode_scores.get("Lose", False))),
                "aborted": bool(interactions.get("Aborted", episode_scores.get("Aborted", False))),
                "main_score": episode_scores.get("Main Score"),
                "quality_score": episode_scores.get("quality_score"),
                "request_count": episode_scores.get("Request Count", interactions.get("Request Count", 0)),
                "parsed_request_count": episode_scores.get(
                    "Parsed Request Count", interactions.get("Parsed Request Count", 0)
                ),
                "violated_request_count": episode_scores.get(
                    "Violated Request Count", interactions.get("Violated Request Count", 0)
                ),
                "parse_error_count": interactions.get("parse_error_count", 0),
                "runtime_error_count": interactions.get("runtime_error_count", 0),
                "internal_consistency_score": interactions.get("internal_consistency_score", 0.0),
                "internal_consistency_violations": interactions.get("internal_consistency_violations", 0),
                "turns_used": interactions.get("turns_used", 0),
                "max_turns": interactions.get("max_turns", experiment.get("max_turns", 0)),
                "efficiency_raw": interactions.get("efficiency_raw", interactions.get("efficiency", 0.0)),
                "tolerance_used": interactions.get("tolerance_used", 0),
                "baseline_kind": instance.get("baseline_kind") or experiment.get("baseline_kind"),
                "n_test_cases_logged": interactions.get("n_test_cases"),
                "instance_path": str(instance_path),
                "scores_path": str(scores_path),
                "experiment_path": str(experiment_path),
                "interactions_path": str(interactions_path),
            }
        )
        if model_filter and row["model"] != model_filter:
            continue
        rows.append(row)

    if not rows:
        if model_filter:
            raise RuntimeError(f"No complete episodes found under: {input_dir} for model `{model_filter}`")
        raise RuntimeError(f"No complete episodes found under: {input_dir}")

    return pd.DataFrame(rows)
