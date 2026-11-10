from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INPUT_PREPROCESSING_ACCESS = {
    "raw": "source_only",
    "trial_z": "sample_local",
    "robust_mad": "full_unlabeled_signal",
    "euclidean_alignment": "full_unlabeled_signal",
    "riemannian_alignment": "full_unlabeled_signal",
}


def trial_channel_zscore(signal: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Normalize every channel using only the current trial."""

    array = np.asarray(signal, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("signal must have shape [channels, time]")
    center = array.mean(axis=-1, keepdims=True)
    scale = array.std(axis=-1, keepdims=True)
    return (array - center) / np.maximum(scale, epsilon)


@dataclass(frozen=True)
class RobustSubjectScale:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, signal: np.ndarray) -> np.ndarray:
        array = np.asarray(signal, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != self.center.shape[0]:
            raise ValueError("signal is incompatible with fitted subject scale")
        return (array - self.center[:, None]) / self.scale[:, None]


def fit_robust_subject_scale(
    signals: np.ndarray,
    epsilon: float = 1e-6,
) -> RobustSubjectScale:
    """Fit channel-wise median/MAD from an unlabeled subject calibration batch."""

    array = np.asarray(signals, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("signals must have shape [trials, channels, time]")
    flattened = array.transpose(1, 0, 2).reshape(array.shape[1], -1)
    center = np.median(flattened, axis=1)
    mad = np.median(np.abs(flattened - center[:, None]), axis=1)
    # 1.4826 makes MAD consistent for a Gaussian standard deviation.
    scale = np.maximum(1.4826 * mad, epsilon)
    return RobustSubjectScale(center=center.astype(np.float32), scale=scale.astype(np.float32))


@dataclass(frozen=True)
class EuclideanAlignment:
    matrix: np.ndarray

    def transform(self, signal: np.ndarray) -> np.ndarray:
        array = np.asarray(signal, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != self.matrix.shape[0]:
            raise ValueError("signal is incompatible with fitted alignment")
        return (self.matrix @ array).astype(np.float32, copy=False)


def _normalized_covariance(signal: np.ndarray, epsilon: float) -> np.ndarray:
    covariance = signal @ signal.T / max(1, signal.shape[-1] - 1)
    trace = float(np.trace(covariance))
    # EEG is stored in volts, so a perfectly valid covariance trace is commonly
    # around 1e-10.  ``epsilon`` is a dimensionless shrinkage strength and must
    # not be reused as an absolute signal-power threshold.  Doing so collapsed
    # every real EEG covariance to I/C while unit-scale synthetic tests passed.
    # Only an actually zero/underflowed trace receives the identity fallback.
    if not np.isfinite(trace) or trace <= np.finfo(np.float64).tiny:
        return np.eye(signal.shape[0], dtype=np.float64) / signal.shape[0]
    return covariance / trace


def fit_euclidean_alignment(
    signals: np.ndarray,
    regularization: float = 1e-5,
) -> EuclideanAlignment:
    """Fit inverse square-root mean covariance from unlabeled target trials."""

    array = np.asarray(signals, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("signals must have shape [trials, channels, time]")
    if not 0.0 < regularization < 1.0:
        raise ValueError("regularization must lie in (0, 1)")
    covariances = [_normalized_covariance(trial, regularization) for trial in array]
    reference = np.mean(covariances, axis=0)
    channels = reference.shape[0]
    reference = (1.0 - regularization) * reference + regularization * (
        np.eye(channels) / channels
    )
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    inverse_root = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, regularization)))) @ (
        eigenvectors.T
    )
    return EuclideanAlignment(matrix=inverse_root.astype(np.float32))


@dataclass(frozen=True)
class RiemannianAlignment:
    matrix: np.ndarray
    iterations: int
    residual: float

    def transform(self, signal: np.ndarray) -> np.ndarray:
        array = np.asarray(signal, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != self.matrix.shape[0]:
            raise ValueError("signal is incompatible with fitted Riemannian alignment")
        return (self.matrix @ array).astype(np.float32, copy=False)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def _spd_power(matrix: np.ndarray, power: float, floor: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(matrix))
    powered = np.power(np.maximum(eigenvalues, floor), power)
    return _symmetrize((eigenvectors * powered) @ eigenvectors.T)


def _spd_log(matrix: np.ndarray, floor: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(matrix))
    logged = np.log(np.maximum(eigenvalues, floor))
    return _symmetrize((eigenvectors * logged) @ eigenvectors.T)


def _spd_exp(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(matrix))
    return _symmetrize((eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T)


def fit_riemannian_alignment(
    signals: np.ndarray,
    regularization: float = 1e-5,
    tolerance: float = 1e-7,
    maximum_iterations: int = 50,
) -> RiemannianAlignment:
    """Fit affine-invariant Riemannian mean covariance using unlabeled trials only.

    This is an explicit Karcher-mean implementation so the journal pipeline does
    not silently change behaviour with a pyRiemann release.  Every trial
    covariance is trace-normalized and shrunk before the mean is estimated.
    """

    array = np.asarray(signals, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError("signals must have shape [trials, channels, time]")
    if not np.isfinite(array).all():
        raise ValueError("signals contain NaN or infinity")
    if not 0.0 < regularization < 1.0:
        raise ValueError("regularization must lie in (0, 1)")
    if tolerance <= 0 or maximum_iterations < 1:
        raise ValueError("invalid Riemannian mean convergence settings")
    channels = array.shape[1]
    identity = np.eye(channels, dtype=np.float64) / channels
    covariances = []
    for trial in array:
        covariance = _normalized_covariance(trial, regularization)
        covariance = (1.0 - regularization) * covariance + regularization * identity
        covariances.append(_symmetrize(covariance))
    reference = _symmetrize(np.mean(covariances, axis=0))
    residual = float("inf")
    completed_iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        root = _spd_power(reference, 0.5, regularization)
        inverse_root = _spd_power(reference, -0.5, regularization)
        tangent = np.mean(
            [_spd_log(inverse_root @ covariance @ inverse_root, regularization) for covariance in covariances],
            axis=0,
        )
        residual = float(np.linalg.norm(tangent, ord="fro"))
        completed_iterations = iteration
        if residual <= tolerance:
            break
        reference = _symmetrize(root @ _spd_exp(tangent) @ root)
    inverse_root = _spd_power(reference, -0.5, regularization)
    return RiemannianAlignment(
        matrix=inverse_root.astype(np.float32),
        iterations=completed_iterations,
        residual=residual,
    )


def fit_input_preprocessor(signals: np.ndarray, mode: str):
    normalized = mode.lower().replace("-", "_")
    if normalized not in INPUT_PREPROCESSING_ACCESS:
        raise KeyError(f"unknown input preprocessing: {mode}")
    if normalized in {"raw", "trial_z"}:
        return None
    if normalized == "robust_mad":
        return fit_robust_subject_scale(signals)
    if normalized == "euclidean_alignment":
        return fit_euclidean_alignment(signals)
    if normalized == "riemannian_alignment":
        return fit_riemannian_alignment(signals)
    raise AssertionError("unreachable preprocessing mode")


def transform_input_signal(signal: np.ndarray, mode: str, fitted=None) -> np.ndarray:
    normalized = mode.lower().replace("-", "_")
    if normalized == "raw":
        return np.asarray(signal, dtype=np.float32)
    if normalized == "trial_z":
        return trial_channel_zscore(signal)
    if normalized not in INPUT_PREPROCESSING_ACCESS:
        raise KeyError(f"unknown input preprocessing: {mode}")
    if fitted is None:
        raise ValueError(f"{normalized} requires an unlabeled subject-fitted transform")
    return fitted.transform(signal)
