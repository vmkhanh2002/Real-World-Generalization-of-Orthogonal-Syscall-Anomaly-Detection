"""Typed models used by the canonical realtime evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _mapping_or_empty(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if hasattr(payload, "to_dict"):
        return dict(payload.to_dict())
    return dict(payload)


def _maybe_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _maybe_str(value: Any) -> str | None:
    return None if value is None else str(value)


@dataclass(slots=True)
class TriggerConfig:
    rolling_mean_prob_gte: float
    rolling_positive_rate_gte: float
    rolling_consecutive_positive_gte: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TriggerConfig":
        return cls(
            rolling_mean_prob_gte=float(payload["rolling_mean_prob_gte"]),
            rolling_positive_rate_gte=float(payload["rolling_positive_rate_gte"]),
            rolling_consecutive_positive_gte=int(payload["rolling_consecutive_positive_gte"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "rolling_mean_prob_gte": self.rolling_mean_prob_gte,
 "rolling_positive_rate_gte": self.rolling_positive_rate_gte,
 "rolling_consecutive_positive_gte": self.rolling_consecutive_positive_gte,
        }


@dataclass(slots=True)
class ClearConfig:
    rolling_mean_prob_lte: float
    rolling_positive_rate_lte: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ClearConfig":
        return cls(
            rolling_mean_prob_lte=float(payload["rolling_mean_prob_lte"]),
            rolling_positive_rate_lte=float(payload["rolling_positive_rate_lte"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "rolling_mean_prob_lte": self.rolling_mean_prob_lte,
 "rolling_positive_rate_lte": self.rolling_positive_rate_lte,
        }


@dataclass(slots=True)
class PolicyConfig:
    policy_name: str
    warmup_windows: int
    rolling_window: int
    phase5_tau: float | None
    trigger: TriggerConfig
    clear: ClearConfig
    mode: str = "streaming"
    input_signal: str = "phase5_prob_t"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PolicyConfig":
        return cls(
            policy_name=str(payload.get("policy_name", "realtime_eval_policy")),
            mode=str(payload.get("mode", "streaming")),
            input_signal=str(payload.get("input_signal", "phase5_prob_t")),
            warmup_windows=int(payload["warmup_windows"]),
            rolling_window=int(payload["rolling_window"]),
            phase5_tau=(
                None if payload.get("phase5_tau") is None else float(payload.get("phase5_tau"))
            ),
            trigger=TriggerConfig.from_mapping(dict(payload.get("trigger") or {})),
            clear=ClearConfig.from_mapping(dict(payload.get("clear") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "policy_name": self.policy_name,
 "mode": self.mode,
 "input_signal": self.input_signal,
 "warmup_windows": self.warmup_windows,
 "rolling_window": self.rolling_window,
 "phase5_tau": self.phase5_tau,
 "trigger": self.trigger.to_dict(),
 "clear": self.clear.to_dict(),
        }


@dataclass(slots=True)
class RealtimeEvalInputs:
    window_scores_jsonl: str
    episodes_json: str
    policy_json: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RealtimeEvalInputs":
        return cls(
            window_scores_jsonl=str(payload["window_scores_jsonl"]),
            episodes_json=str(payload["episodes_json"]),
            policy_json=str(payload["policy_json"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "window_scores_jsonl": self.window_scores_jsonl,
 "episodes_json": self.episodes_json,
 "policy_json": self.policy_json,
        }


@dataclass(slots=True)
class AttackFamilyMetrics:
    attack_episodes: int | None = None
    episode_recall: float | None = None
    recall_within_5_windows: float | None = None
    recall_within_10_windows: float | None = None
    recall_within_20_windows: float | None = None
    median_detection_delay_windows: float | None = None
    p90_detection_delay_windows: float | None = None
    false_alert_before_attack_rate: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AttackFamilyMetrics":
        return cls(
            attack_episodes=_maybe_int(payload.get("attack_episodes")),
            episode_recall=_maybe_float(payload.get("episode_recall")),
            recall_within_5_windows=_maybe_float(payload.get("recall_within_5_windows")),
            recall_within_10_windows=_maybe_float(payload.get("recall_within_10_windows")),
            recall_within_20_windows=_maybe_float(payload.get("recall_within_20_windows")),
            median_detection_delay_windows=_maybe_float(payload.get("median_detection_delay_windows")),
            p90_detection_delay_windows=_maybe_float(payload.get("p90_detection_delay_windows")),
            false_alert_before_attack_rate=_maybe_float(payload.get("false_alert_before_attack_rate")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "attack_episodes": self.attack_episodes,
 "episode_recall": self.episode_recall,
 "recall_within_5_windows": self.recall_within_5_windows,
 "recall_within_10_windows": self.recall_within_10_windows,
 "recall_within_20_windows": self.recall_within_20_windows,
 "median_detection_delay_windows": self.median_detection_delay_windows,
 "p90_detection_delay_windows": self.p90_detection_delay_windows,
 "false_alert_before_attack_rate": self.false_alert_before_attack_rate,
        }


@dataclass(slots=True)
class RealtimeEvalSummary:
    episodes_total_input: int
    episodes_evaluated: int
    episodes_missing_scores: int
    episodes_with_n_window_mismatch: int
    attack_episodes: int
    benign_episodes: int
    episode_recall: float | None
    median_detection_delay_windows: float | None
    p90_detection_delay_windows: float | None
    false_alert_rate_per_benign_episode: float | None
    false_alert_windows_per_1000_benign_windows: float | None
    alert_precision_episode_level: float | None
    alert_precision_window_level: float | None
    alert_recall_window_level: float | None
    mean_alert_duration_windows: float | None
    alert_flicker_rate_per_episode: float | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RealtimeEvalSummary":
        def maybe_float(name: str) -> float | None:
            value = payload.get(name)
            return None if value is None else float(value)

        return cls(
            episodes_total_input=int(payload["episodes_total_input"]),
            episodes_evaluated=int(payload["episodes_evaluated"]),
            episodes_missing_scores=int(payload["episodes_missing_scores"]),
            episodes_with_n_window_mismatch=int(payload["episodes_with_n_window_mismatch"]),
            attack_episodes=int(payload["attack_episodes"]),
            benign_episodes=int(payload["benign_episodes"]),
            episode_recall=maybe_float("episode_recall"),
            median_detection_delay_windows=maybe_float("median_detection_delay_windows"),
            p90_detection_delay_windows=maybe_float("p90_detection_delay_windows"),
            false_alert_rate_per_benign_episode=maybe_float("false_alert_rate_per_benign_episode"),
            false_alert_windows_per_1000_benign_windows=maybe_float("false_alert_windows_per_1000_benign_windows"),
            alert_precision_episode_level=maybe_float("alert_precision_episode_level"),
            alert_precision_window_level=maybe_float("alert_precision_window_level"),
            alert_recall_window_level=maybe_float("alert_recall_window_level"),
            mean_alert_duration_windows=maybe_float("mean_alert_duration_windows"),
            alert_flicker_rate_per_episode=maybe_float("alert_flicker_rate_per_episode"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "episodes_total_input": self.episodes_total_input,
 "episodes_evaluated": self.episodes_evaluated,
 "episodes_missing_scores": self.episodes_missing_scores,
 "episodes_with_n_window_mismatch": self.episodes_with_n_window_mismatch,
 "attack_episodes": self.attack_episodes,
 "benign_episodes": self.benign_episodes,
 "episode_recall": self.episode_recall,
 "median_detection_delay_windows": self.median_detection_delay_windows,
 "p90_detection_delay_windows": self.p90_detection_delay_windows,
 "false_alert_rate_per_benign_episode": self.false_alert_rate_per_benign_episode,
 "false_alert_windows_per_1000_benign_windows": self.false_alert_windows_per_1000_benign_windows,
 "alert_precision_episode_level": self.alert_precision_episode_level,
 "alert_precision_window_level": self.alert_precision_window_level,
 "alert_recall_window_level": self.alert_recall_window_level,
 "mean_alert_duration_windows": self.mean_alert_duration_windows,
 "alert_flicker_rate_per_episode": self.alert_flicker_rate_per_episode,
        }


@dataclass(slots=True)
class AlertSegment:
    start_window: int
    end_window: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AlertSegment":
        return cls(
            start_window=int(payload["start_window"]),
            end_window=int(payload["end_window"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "start_window": self.start_window,
 "end_window": self.end_window,
        }


@dataclass(slots=True)
class EpisodeEvaluationResult:
    episode_id: str
    score_file: str
    attack_family: str | None
    intended_label: str
    n_windows: int
    has_attack_segment: bool
    has_attack_window_range: bool
    attack_start_window: int | None
    attack_end_window: int | None
    detected: bool
    first_alert_window: int | None
    first_alert_inside_attack_window: int | None
    detection_delay_windows: int | None
    detected_within_5_windows: bool
    detected_within_10_windows: bool
    detected_within_20_windows: bool
    alert_time_inside_attack: int
    alert_time_before_attack: int
    alert_time_after_attack: int
    false_alert_before_attack: bool
    had_false_alert: bool
    first_false_alert_window: int | None
    false_alert_count: int
    false_alert_duration_windows: int
    alert_duration_windows_total: int
    alert_segment_count: int
    alert_flicker_count: int
    recovery_delay_after_attack_end: int | None
    policy_phase5_tau: float | None
    alert_segments: list[AlertSegment]
    window_true_alert_count: int
    window_alert_count: int
    window_attack_count: int
    declared_n_windows: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EpisodeEvaluationResult":
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            score_file=str(payload.get("score_file", "")),
            attack_family=_maybe_str(payload.get("attack_family")),
            intended_label=str(payload.get("intended_label", "")),
            n_windows=int(payload.get("n_windows", 0)),
            declared_n_windows=_maybe_int(payload.get("declared_n_windows")),
            has_attack_segment=bool(payload.get("has_attack_segment", False)),
            has_attack_window_range=bool(payload.get("has_attack_window_range", False)),
            attack_start_window=_maybe_int(payload.get("attack_start_window")),
            attack_end_window=_maybe_int(payload.get("attack_end_window")),
            detected=bool(payload.get("detected", False)),
            first_alert_window=_maybe_int(payload.get("first_alert_window")),
            first_alert_inside_attack_window=_maybe_int(payload.get("first_alert_inside_attack_window")),
            detection_delay_windows=_maybe_int(payload.get("detection_delay_windows")),
            detected_within_5_windows=bool(payload.get("detected_within_5_windows", False)),
            detected_within_10_windows=bool(payload.get("detected_within_10_windows", False)),
            detected_within_20_windows=bool(payload.get("detected_within_20_windows", False)),
            alert_time_inside_attack=int(payload.get("alert_time_inside_attack", 0)),
            alert_time_before_attack=int(payload.get("alert_time_before_attack", 0)),
            alert_time_after_attack=int(payload.get("alert_time_after_attack", 0)),
            false_alert_before_attack=bool(payload.get("false_alert_before_attack", False)),
            had_false_alert=bool(payload.get("had_false_alert", False)),
            first_false_alert_window=_maybe_int(payload.get("first_false_alert_window")),
            false_alert_count=int(payload.get("false_alert_count", 0)),
            false_alert_duration_windows=int(payload.get("false_alert_duration_windows", 0)),
            alert_duration_windows_total=int(payload.get("alert_duration_windows_total", 0)),
            alert_segment_count=int(payload.get("alert_segment_count", 0)),
            alert_flicker_count=int(payload.get("alert_flicker_count", 0)),
            recovery_delay_after_attack_end=_maybe_int(payload.get("recovery_delay_after_attack_end")),
            policy_phase5_tau=_maybe_float(payload.get("policy_phase5_tau")),
            alert_segments=[
                AlertSegment.from_mapping(_mapping_or_empty(item))
                for item in list(payload.get("alert_segments") or [])
            ],
            window_true_alert_count=int(payload.get("window_true_alert_count", 0)),
            window_alert_count=int(payload.get("window_alert_count", 0)),
            window_attack_count=int(payload.get("window_attack_count", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "episode_id": self.episode_id,
 "score_file": self.score_file,
 "attack_family": self.attack_family,
 "intended_label": self.intended_label,
 "n_windows": self.n_windows,
 "declared_n_windows": self.declared_n_windows,
 "has_attack_segment": self.has_attack_segment,
 "has_attack_window_range": self.has_attack_window_range,
 "attack_start_window": self.attack_start_window,
 "attack_end_window": self.attack_end_window,
 "detected": self.detected,
 "first_alert_window": self.first_alert_window,
 "first_alert_inside_attack_window": self.first_alert_inside_attack_window,
 "detection_delay_windows": self.detection_delay_windows,
 "detected_within_5_windows": self.detected_within_5_windows,
 "detected_within_10_windows": self.detected_within_10_windows,
 "detected_within_20_windows": self.detected_within_20_windows,
 "alert_time_inside_attack": self.alert_time_inside_attack,
 "alert_time_before_attack": self.alert_time_before_attack,
 "alert_time_after_attack": self.alert_time_after_attack,
 "false_alert_before_attack": self.false_alert_before_attack,
 "had_false_alert": self.had_false_alert,
 "first_false_alert_window": self.first_false_alert_window,
 "false_alert_count": self.false_alert_count,
 "false_alert_duration_windows": self.false_alert_duration_windows,
 "alert_duration_windows_total": self.alert_duration_windows_total,
 "alert_segment_count": self.alert_segment_count,
 "alert_flicker_count": self.alert_flicker_count,
 "recovery_delay_after_attack_end": self.recovery_delay_after_attack_end,
 "policy_phase5_tau": self.policy_phase5_tau,
 "alert_segments": [segment.to_dict() for segment in self.alert_segments],
 "window_true_alert_count": self.window_true_alert_count,
 "window_alert_count": self.window_alert_count,
 "window_attack_count": self.window_attack_count,
        }


@dataclass(slots=True)
class TopFailureExample:
    episode_id: str
    attack_family: str | None
    detection_delay_windows: int | None
    first_alert_window: int | None
    false_alert_duration_windows: int | None
    score_file: str | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TopFailureExample":
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            attack_family=_maybe_str(payload.get("attack_family")),
            detection_delay_windows=_maybe_int(payload.get("detection_delay_windows")),
            first_alert_window=_maybe_int(payload.get("first_alert_window")),
            false_alert_duration_windows=_maybe_int(payload.get("false_alert_duration_windows")),
            score_file=_maybe_str(payload.get("score_file")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "episode_id": self.episode_id,
 "attack_family": self.attack_family,
 "detection_delay_windows": self.detection_delay_windows,
 "first_alert_window": self.first_alert_window,
 "false_alert_duration_windows": self.false_alert_duration_windows,
 "score_file": self.score_file,
        }


@dataclass(slots=True)
class TopFailuresReport:
    missed_attacks: list[TopFailureExample] = field(default_factory=list)
    slow_detected_attacks: list[TopFailureExample] = field(default_factory=list)
    benign_false_alert_examples: list[TopFailureExample] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TopFailuresReport":
        return cls(
            missed_attacks=[
                TopFailureExample.from_mapping(_mapping_or_empty(item))
                for item in list(payload.get("missed_attacks") or [])
            ],
            slow_detected_attacks=[
                TopFailureExample.from_mapping(_mapping_or_empty(item))
                for item in list(payload.get("slow_detected_attacks") or [])
            ],
            benign_false_alert_examples=[
                TopFailureExample.from_mapping(_mapping_or_empty(item))
                for item in list(payload.get("benign_false_alert_examples") or [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "missed_attacks": [item.to_dict() for item in self.missed_attacks],
 "slow_detected_attacks": [item.to_dict() for item in self.slow_detected_attacks],
 "benign_false_alert_examples": [item.to_dict() for item in self.benign_false_alert_examples],
        }


@dataclass(slots=True)
class MissingScoreRow:
    episode_id: str | None
    score_file: str | None
    source_file: str | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MissingScoreRow":
        return cls(
            episode_id=_maybe_str(payload.get("episode_id")),
            score_file=_maybe_str(payload.get("score_file")),
            source_file=_maybe_str(payload.get("source_file")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
 "episode_id": self.episode_id,
 "score_file": self.score_file,
 "source_file": self.source_file,
        }


def _top_failures_from_any(payload: Any) -> TopFailuresReport:
    if isinstance(payload, TopFailuresReport):
        return payload
    if payload is None:
        return TopFailuresReport()
    if not isinstance(payload, Mapping) and not hasattr(payload, "to_dict"):
        return TopFailuresReport()
    return TopFailuresReport.from_mapping(_mapping_or_empty(payload))


def _missing_score_rows_from_any(values: Any) -> list[MissingScoreRow]:
    return [MissingScoreRow.from_mapping(_mapping_or_empty(item)) for item in list(values or [])]


def _episode_results_from_any(values: Any) -> list[EpisodeEvaluationResult]:
    return [EpisodeEvaluationResult.from_mapping(_mapping_or_empty(item)) for item in list(values or [])]


@dataclass(slots=True)
class RealtimeEvalReport:
    schema_version: int
    generated_at_utc: str
    policy: PolicyConfig
    inputs: RealtimeEvalInputs
    summary: RealtimeEvalSummary
    family_metrics: dict[str, AttackFamilyMetrics]
    top_failures: TopFailuresReport
    missing_score_rows: list[MissingScoreRow]
    episode_results: list[EpisodeEvaluationResult] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RealtimeEvalReport":
        family_metrics = {
            str(name): AttackFamilyMetrics.from_mapping(_mapping_or_empty(value))
            for name, value in _mapping_or_empty(payload.get("family_metrics")).items()
        }
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            generated_at_utc=str(payload["generated_at_utc"]),
            policy=PolicyConfig.from_mapping(_mapping_or_empty(payload["policy"])),
            inputs=RealtimeEvalInputs.from_mapping(_mapping_or_empty(payload["inputs"])),
            summary=RealtimeEvalSummary.from_mapping(_mapping_or_empty(payload["summary"])),
            family_metrics=family_metrics,
            top_failures=_top_failures_from_any(payload.get("top_failures")),
            missing_score_rows=_missing_score_rows_from_any(payload.get("missing_score_rows")),
            episode_results=(
                None
                if payload.get("episode_results") is None
                else _episode_results_from_any(payload.get("episode_results"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
 "schema_version": self.schema_version,
 "generated_at_utc": self.generated_at_utc,
 "policy": self.policy.to_dict(),
 "inputs": self.inputs.to_dict(),
 "summary": self.summary.to_dict(),
 "family_metrics": {
                name: AttackFamilyMetrics.from_mapping(_mapping_or_empty(value)).to_dict()
                for name, value in self.family_metrics.items()
            },
 "top_failures": _top_failures_from_any(self.top_failures).to_dict(),
 "missing_score_rows": [item.to_dict() for item in _missing_score_rows_from_any(self.missing_score_rows)],
        }
        if self.episode_results is not None:
            payload["episode_results"] = [item.to_dict() for item in _episode_results_from_any(self.episode_results)]
        return payload
