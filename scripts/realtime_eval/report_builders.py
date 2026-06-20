"""Realtime-only report builders used by the evaluation CLI."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_local_module(module_name: str):
    module_path = SCRIPT_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"realtime_eval_local_{module_name}", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    from realtime_eval import models  # type: ignore # pylint: disable=import error

    if Path(getattr(models, "__file__", "")).resolve() != (SCRIPT_DIR / "models.py").resolve():
        raise ImportError
except ImportError:
    models = _load_local_module("models")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value)


def _episode_results(values: Sequence[Any] | None) -> list[models.EpisodeEvaluationResult]:
    out: list[models.EpisodeEvaluationResult] = []
    for item in list(values or []):
        if isinstance(item, models.EpisodeEvaluationResult):
            out.append(item)
        else:
            out.append(models.EpisodeEvaluationResult.from_mapping(_mapping(item)))
    return out


def _missing_score_rows(values: Sequence[Any] | None) -> list[models.MissingScoreRow]:
    out: list[models.MissingScoreRow] = []
    for item in list(values or []):
        if isinstance(item, models.MissingScoreRow):
            out.append(item)
        else:
            out.append(models.MissingScoreRow.from_mapping(_mapping(item)))
    return out


def _build_family_metrics(
    *,
    rt_module: Any,
    episode_results: Sequence[models.EpisodeEvaluationResult],
) -> dict[str, models.AttackFamilyMetrics]:
    raw_metrics = dict(rt_module.build_attack_family_metrics(episode_results))
    return {
        str(name): (
            value
            if isinstance(value, models.AttackFamilyMetrics)
            else models.AttackFamilyMetrics.from_mapping(_mapping(value))
        )
        for name, value in raw_metrics.items()
    }


def _build_top_failures(
    *,
    rt_module: Any,
    episode_results: Sequence[models.EpisodeEvaluationResult],
    top_n: int,
) -> models.TopFailuresReport:
    raw = rt_module.build_top_failures_report(episode_results, top_n)
    if isinstance(raw, models.TopFailuresReport):
        return raw
    return models.TopFailuresReport.from_mapping(_mapping(raw))


def build_realtime_summary(
    *,
    rt_module: Any,
    episode_results: Sequence[models.EpisodeEvaluationResult],
    episodes_total_input: int,
    episodes_missing_scores: int,
    episodes_with_n_window_mismatch: int,
) -> models.RealtimeEvalSummary:
    rows = list(episode_results)
    attack_rows = [row for row in rows if row.has_attack_segment]
    benign_rows = [row for row in rows if not row.has_attack_segment]
    detected_attack_rows = [row for row in attack_rows if row.detected]
    delays = [
        int(row.detection_delay_windows)
        for row in detected_attack_rows
        if row.detection_delay_windows is not None
    ]
    benign_false_alert_rows = [row for row in benign_rows if row.had_false_alert]
    positive_episode_rows = [row for row in rows if row.window_alert_count > 0]
    total_alert_windows = int(sum(row.window_alert_count for row in rows))
    total_true_alert_windows = int(sum(row.window_true_alert_count for row in rows))
    total_attack_windows = int(sum(row.window_attack_count for row in rows))
    total_benign_windows = int(sum(row.n_windows for row in benign_rows))
    total_benign_false_alert_windows = int(sum(row.window_alert_count for row in benign_rows))

    return models.RealtimeEvalSummary(
        episodes_total_input=int(episodes_total_input),
        episodes_evaluated=len(rows),
        episodes_missing_scores=int(episodes_missing_scores),
        episodes_with_n_window_mismatch=int(episodes_with_n_window_mismatch),
        attack_episodes=len(attack_rows),
        benign_episodes=len(benign_rows),
        episode_recall=rt_module.safe_div(len(detected_attack_rows), len(attack_rows)),
        median_detection_delay_windows=(float(rt_module.median(delays)) if delays else None),
        p90_detection_delay_windows=rt_module.percentile(delays, 90.0),
        false_alert_rate_per_benign_episode=rt_module.safe_div(
            len(benign_false_alert_rows),
            len(benign_rows),
        ),
        false_alert_windows_per_1000_benign_windows=(
            (1000.0 * total_benign_false_alert_windows / total_benign_windows)
            if total_benign_windows > 0
            else None
        ),
        alert_precision_episode_level=rt_module.safe_div(
            len(detected_attack_rows),
            len(positive_episode_rows),
        ),
        alert_precision_window_level=rt_module.safe_div(
            total_true_alert_windows,
            total_alert_windows,
        ),
        alert_recall_window_level=rt_module.safe_div(
            total_true_alert_windows,
            total_attack_windows,
        ),
        mean_alert_duration_windows=rt_module.safe_div(
            sum(row.alert_duration_windows_total for row in rows),
            sum(row.alert_segment_count for row in rows),
        ),
        alert_flicker_rate_per_episode=rt_module.safe_div(
            sum(row.alert_flicker_count for row in rows),
            len(rows),
        ),
    )


def build_realtime_eval_report(
    *,
    rt_module: Any,
    generated_at_utc: str,
    policy: Any,
    inputs: Any,
    episode_results: Sequence[Any],
    episodes_total_input: int,
    missing_score_rows: Sequence[Any] | None,
    episodes_with_n_window_mismatch: int,
    top_n_failures: int,
    omit_episode_results: bool,
) -> models.RealtimeEvalReport:
    rows = _episode_results(episode_results)
    missing_rows = _missing_score_rows(missing_score_rows)
    return models.RealtimeEvalReport(
        schema_version=1,
        generated_at_utc=str(generated_at_utc),
        policy=(
            policy
            if isinstance(policy, models.PolicyConfig)
            else models.PolicyConfig.from_mapping(_mapping(policy))
        ),
        inputs=(
            inputs
            if isinstance(inputs, models.RealtimeEvalInputs)
            else models.RealtimeEvalInputs.from_mapping(_mapping(inputs))
        ),
        summary=build_realtime_summary(
            rt_module=rt_module,
            episode_results=rows,
            episodes_total_input=episodes_total_input,
            episodes_missing_scores=len(missing_rows),
            episodes_with_n_window_mismatch=episodes_with_n_window_mismatch,
        ),
        family_metrics=_build_family_metrics(
            rt_module=rt_module,
            episode_results=rows,
        ),
        top_failures=_build_top_failures(
            rt_module=rt_module,
            episode_results=rows,
            top_n=max(1, int(top_n_failures)),
        ),
        missing_score_rows=missing_rows,
        episode_results=None if omit_episode_results else rows,
    )
