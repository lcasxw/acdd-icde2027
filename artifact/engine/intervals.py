import hashlib
import numpy as np
SEED=20260906
SAMPLES=10000

def seed_for(key: str) -> int:
    return (SEED + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)) % (2**32)

def interval(values: list[float], key: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    assert len(array) > 1 and np.isfinite(array).all()
    rng = np.random.default_rng(seed_for(key))
    draws = array[rng.integers(0, len(array), (SAMPLES, len(array)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975], method="linear"))
