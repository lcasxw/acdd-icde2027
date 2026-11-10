from collections import defaultdict
from statistics import mean
from typing import Any, Iterable
import numpy as np

def cluster_seed(rows: Iterable[dict[str, Any]], value: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["protocol_id"]), str(row["participant_id"]))].append(float(row[value]))
    return [{"protocol_id": key[0], "participant_id": key[1], value: mean(values)} for key, values in sorted(grouped.items())]

def three_weight_bootstrap(rows: list[dict[str, Any]], value: str, samples: int, seed: int) -> dict[str, Any]:
    clustered = cluster_seed(rows, value)
    protocols = sorted({row["protocol_id"] for row in clustered})
    participants = sorted({row["participant_id"] for row in clustered})
    lookup = {(row["protocol_id"], row["participant_id"]): float(row[value]) for row in clustered}
    matrix = np.full((len(protocols), len(participants)), np.nan, dtype=np.float64)
    for i, protocol in enumerate(protocols):
        for j, participant in enumerate(participants):
            if (protocol, participant) in lookup:
                matrix[i, j] = lookup[(protocol, participant)]
    participant_values = np.nanmean(matrix, axis=0)
    protocol_values = np.nanmean(matrix, axis=1)
    rng = np.random.default_rng(seed)
    participant_draws = np.mean(participant_values[rng.integers(0, len(participant_values), size=(samples, len(participant_values)))], axis=1)
    protocol_draws = np.mean(protocol_values[rng.integers(0, len(protocol_values), size=(samples, len(protocol_values)))], axis=1)

    def summary(point: float, draws: np.ndarray) -> dict[str, float]:
        return {"point": float(point), "ci_lower": float(np.quantile(draws, 0.025, method="linear")), "ci_upper": float(np.quantile(draws, 0.975, method="linear"))}

    participant = summary(float(np.nanmean(participant_values)), participant_draws)
    return {
        "participant_weighted": participant,
        "dataset_balanced": participant,
        "protocol_balanced": summary(float(np.nanmean(protocol_values)), protocol_draws),
    }

def rebuild_crossfit(rows: list[dict[str, str]], samples: int, seed: int) -> list[dict[str, Any]]:
    output = []
    for backbone in sorted({row["backbone"] for row in rows}):
        members = [row for row in rows if row["backbone"] == backbone]
        for metric in ("cross_fitted_headroom", "cross_fitted_decision_loss", "in_sample_headroom", "in_sample_decision_loss"):
            estimates = three_weight_bootstrap(members, metric, samples, seed + len(output))
            for weighting, result in estimates.items():
                output.append({"backbone": backbone, "metric": metric, "weighting": weighting, **result})
    return output
