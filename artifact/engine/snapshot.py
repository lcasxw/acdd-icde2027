from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from acdd import strengthening, threshold_policies as policies, threshold_policies_runtime_v2 as runtime
from acdd.source_calibration import sigmoid, apply_calibrator as frozen_apply

@dataclass(frozen=True)
class SnapshotPlatt:
    slope: float
    intercept: float
    source_center: float
    centered_intercept: float
    kind: str = "platt"

def load_model(record):
    names = ("slope", "intercept", "source_center", "centered_intercept")
    values = {name: float.fromhex(record[name + "_hex"]) for name in names}
    if any(values[name] != record[name] for name in names):
        raise ValueError("Snapshot decimal/hex disagreement")
    if record["status"] != "source_fit_verified" or not record["feasible"] or record["kkt_residual"] >= 1e-10:
        raise ValueError("Unverified source-only snapshot")
    return SnapshotPlatt(**values)

def apply_snapshot(model, raw_scores):
    if not isinstance(model, SnapshotPlatt):
        return frozen_apply(model, raw_scores)
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Finite one-dimensional score vector required")
    return sigmoid(model.slope * (scores - model.source_center) + model.centered_intercept)

@contextmanager
def shared_snapshot(model, source_scores, source_labels):
    expected = (tuple(map(float, source_scores)), tuple(map(int, source_labels)))
    old_cached, old_apply, old_runtime_apply = policies._cached_calibrator, policies.apply_calibrator, runtime.apply_calibrator
    old_prequential = policies.ThresholdPolicy.predict_prequential_sequence

    def get_model(raw_logits, labels, kind):
        if kind != "platt":
            return old_cached(raw_logits, labels, kind)
        if raw_logits != expected[0] or labels != expected[1]:
            raise ValueError("Snapshot was requested for different source validation data")
        return model

    policies._cached_calibrator = get_model
    policies.apply_calibrator = apply_snapshot
    runtime.apply_calibrator = apply_snapshot
    runtime.install_runtime_v2_patch()
    try:
        yield
    finally:
        policies._cached_calibrator, policies.apply_calibrator = old_cached, old_apply
        runtime.apply_calibrator = old_runtime_apply
        policies.ThresholdPolicy.predict_prequential_sequence = old_prequential

class Recorder:
    def __init__(self, policy, decisions, prior_cutoff=False):
        self.policy, self.decisions, self.prior_cutoff = policy, decisions, prior_cutoff

    def predict(self, value):
        prediction = (int(apply_snapshot(self.policy.calibrator, [value])[0] >= self.policy.source_prior)
                      if self.prior_cutoff else self.policy.predict(value))
        self.decisions.append(prediction)
        return prediction

    def predict_prequential_sequence(self, *args, **kwargs):
        if self.prior_cutoff:
            raise ValueError("Paired utility diagnostic is protected-endpoint only")
        values = self.policy.predict_prequential_sequence(*args, **kwargs)
        self.decisions.extend(values)
        return values

def metric_with_record(trials, scores, labels, model, method, endpoint, ridge, prior_cutoff=False):
    decisions = []
    old_fit = strengthening.fit

    def fitted(*args, **kwargs):
        policy = old_fit(*args, **kwargs)
        if method in {"source_platt", "platt_em", "rlls_bbse"} and policy.calibrator is not model:
            raise ValueError("Related policy did not consume the identical snapshot")
        return Recorder(policy, decisions, prior_cutoff)

    strengthening.fit = fitted
    try:
        metric = strengthening.policy_endpoint_metrics(trials, scores, labels, method=method,
                    endpoint=endpoint, prefix_length=10, rlls_ridge=ridge)
    finally:
        strengthening.fit = old_fit
    if len(decisions) != metric["n_trials"]:
        raise ValueError("Decision count does not match endpoint")
    import hashlib
    metric["decisions_uint8_sha256"] = hashlib.sha256(bytes(decisions)).hexdigest()
    return metric
