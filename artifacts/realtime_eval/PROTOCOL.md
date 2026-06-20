# Real-Time Evaluation Protocol Draft

## Goal

Evaluate the hybrid detector as an online alerting system, not as an offline file classifier.

The current offline logic answers:
- "At the end of the file, was the final verdict correct?"

The real-time logic should answer:
- "Did the system alert early enough?"
- "Did it alert too often on benign behavior?"
- "Was the alert stable or noisy?"

## Why File-Level Evaluation Is Not Enough

The current detector aggregates the whole file using:
- `phase5_prob_mean`
- `phase5_positive_rate`
- `phase5_max_consecutive_positive_windows`

These are full-episode statistics. In streaming mode, they are not known in advance.

In a real-time stream:
- the prefix seen at time `t` is incomplete
- the final mean probability is a moving target
- a strong early signal can later be diluted by benign windows
- a correct final file verdict may still be operationally useless if the alert comes too late

## Unit of Evaluation

Offline:
- one file -> one final decision

Real-time:
- one stream episode -> one alert timeline

An episode may contain:
- benign-only traffic
- benign prefix + attack segment + benign suffix
- multiple attack bursts

## Proposed Data Model

Each episode should contain:
- `episode_id`
- `source_file`
- `window_size`
- `stride`
- `n_windows`
- `ground_truth_segments`

Each ground truth segment should contain:
- `label`
- `start_window`
- `end_window`

Recommended labels:
- `benign`
- `attack`

Optional:
- `attack_family`
- `attack_subfamily`
- `notes`

## Online Detector State

At each window `t`, the detector should expose:
- `path_a_score_t`
- `path_b_score_t`
- `path_c_score_t`
- `phase5_prob_t`
- `phase5_positive_t = (phase5_prob_t >= tau)`

The online evaluator should then apply an alert policy on the prefix `1..t`.

## Draft Online Alert Policy

We should not reuse the exact offline file rule as-is.

Instead, use rolling statistics:
- `rolling_mean_prob`
- `rolling_positive_rate`
- `rolling_max_consecutive_positive`

Recommended first policy:

Parameters:
- `warmup_windows = 5`
- `rolling_window = 20`
- `alert_mean_prob = 0.60`
- `alert_positive_rate = 0.60`
- `alert_consecutive_positive = 3`
- `clear_mean_prob = 0.45`
- `clear_positive_rate = 0.30`
- `clear_consecutive_positive = 1`

Trigger rule:
- after `warmup_windows`, raise alert if:
  - `rolling_mean_prob >= alert_mean_prob`
  - and `rolling_positive_rate >= alert_positive_rate`
  - and `rolling_max_consecutive_positive >= alert_consecutive_positive`

Clear rule:
- clear alert only if both:
  - `rolling_mean_prob <= clear_mean_prob`
  - `rolling_positive_rate <= clear_positive_rate`

Rationale:
- this creates hysteresis
- avoids flicker
- is closer to operational alerting than end-of-file averaging

## Required Metrics

### Episode-Level Metrics

For attack episodes:
- `detected`: whether any alert overlaps the attack segment
- `first_alert_window`
- `detection_delay_windows`
- `detected_within_5_windows`
- `detected_within_10_windows`
- `detected_within_20_windows`
- `alert_time_inside_attack`
- `alert_time_before_attack`
- `alert_time_after_attack`

For benign-only episodes:
- `had_false_alert`
- `first_false_alert_window`
- `false_alert_count`
- `false_alert_duration_windows`

### Aggregate Metrics

Primary:
- `episode_recall`
- `median_detection_delay_windows`
- `p90_detection_delay_windows`
- `false_alert_rate_per_benign_episode`
- `false_alert_windows_per_1000_benign_windows`

Secondary:
- `alert_precision_episode_level`
- `alert_precision_window_level`
- `mean_alert_duration`
- `alert_flicker_rate`
- `recovery_delay_after_attack_end`

## Suggested Evaluation Views

### 1. Real-Time Detection Performance

By attack family:
- recall within 5 windows
- recall within 10 windows
- recall within 20 windows
- median delay

### 2. Benign Stability

By benign stream:
- false alert count
- false alert duration
- flicker count

### 3. Policy Tradeoff Curves

Sweep:
- alert threshold
- rolling window length
- consecutive-positive requirement

Plot:
- delay vs false alarm rate
- recall vs false alarm rate

## Benchmark Scenarios

### Scenario A: Attack-Only Replay

Use current attack files as pure attack episodes.

Pros:
- easy to build

Cons:
- unrealistic start condition
- no benign prefix

### Scenario B: Benign Prefix + Attack Tail

Construct a synthetic online episode:
- benign windows first
- then attack file windows

Pros:
- much closer to deployment
- supports time-to-detect metrics

Cons:
- requires a compositing script

### Scenario C: Benign Prefix + Attack + Benign Recovery

Construct:
- benign prefix
- attack segment
- benign suffix

Pros:
- supports detection delay
- supports false alert before onset
- supports recovery delay after attack end

This is the recommended benchmark design.

## Proposed Report Sections

The evaluator report should contain:

1. Policy configuration
2. Overall real-time metrics
3. Family-level detection delay table
4. Benign false alert table
5. Top failure cases
6. Alert timeline examples

## Acceptance Criteria For First Real-Time Version

The first version is useful if it can:
- read episode schemas with attack segments
- consume per-window phase5 probabilities
- generate alert timelines
- compute detection delay and false alert metrics
- break down results by attack family

## Recommended Implementation Path

Phase 1:
- expose per-window `phase5_prob_t`
- define episode schema
- implement one rolling alert policy

Phase 2:
- add attack onset benchmarks
- add family-level delay analysis
- add threshold sweeps

Phase 3:
- compare multiple alert policies
- calibrate for different operational budgets

## Suggested Future Folder Layout If We Implement It

This draft does not create code yet, but a likely implementation shape is:

- `scripts/realtime_eval/export_window_scores.py`
- `scripts/realtime_eval/evaluate_realtime_detect.py`
- `scripts/realtime_eval/build_streaming_episodes.py`
- `tests/realtime_eval/`

