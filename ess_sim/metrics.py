"""Opinion-dynamics metrics: the three dependent variables (mean concern, dispersion, bimodality)."""
import numpy as np

from scipy.stats import skew, kurtosis


def _active(concerns, stubborn):
    """Concern values of non-stubborn citizens with valid concern."""
    return [c for u, c in concerns.items()
            if c is not None and not stubborn.get(u, False)]


def metric_dispersion(concerns, stubborn):
    # Bramson dispersion sense: population SD of concern over non-stubborn citizens.
    v = _active(concerns, stubborn)
    return float(np.std(v)) if len(v) > 1 else 0.0


def metric_mean_concern(concerns, stubborn):
    # mean concern over non-stubborn citizens -- quantifies upward drift.
    v = _active(concerns, stubborn)
    return float(np.mean(v)) if v else float("nan")


def metric_bimodality(concerns, stubborn):
    # Bramson regionalization: Sarle's bimodality coefficient; > ~0.555 suggests bimodal.
    x = np.asarray(_active(concerns, stubborn), dtype=float)
    n = len(x)
    if n < 4:
        return float("nan")
    if np.std(x) == 0:            # zero variance: skew/kurtosis undefined, avoid RuntimeWarning
        return float("nan")
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=True, bias=False)
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g ** 2 + 1) / denom) if denom else float("nan")