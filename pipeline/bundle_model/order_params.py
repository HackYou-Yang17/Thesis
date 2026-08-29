"""
orderparams.py -- orientational order parameters that read ZERO for isotropic input.

Why the family split had to go
------------------------------
The previous metric picked a director, split fibres at +-45 deg from it, and
reported nematic = |H-V|/(H+V), quartic = 2min(H,V)/(H+V), plus the angle
between the two family means. Fed PURE RANDOM angles that returns:

    n= 12  interfamily 84.1 +- 6.3 deg,  quartic -> ~0.8
    n=200  interfamily 89.6 +- 0.3 deg,  quartic -> ~1.0

i.e. an isotropic network scores as near-perfect orthogonal order, and scores
BETTER the more fibres it has. The split is at +-45 deg, so the two halves are
~90 deg apart by construction whatever the data does.

The replacement
---------------
Standard circular order parameters at two harmonics:

    C2 = |<exp(2i.theta)>|     nematic  (one axis)
    C4 = |<exp(4i.theta)>|     tetratic (two orthogonal axes)

                    C2      C4
    isotropic        0       0
    one family       1       1
    two at 90 deg    0       1
    two at 60 deg   mid     low

So C4 high with C2 low is the quartic signature, and isotropic is separable
from both -- which the family split could not do.

Finite-n bias is removed exactly. For k random angles E[|<exp(ik.theta)>|^2]
= 1/n, so the unbiased squared estimator is (n|m|^2 - 1)/(n - 1), floored at 0.
With weights, n is the effective count (sum w)^2 / sum w^2.

Interfamily angle
-----------------
Estimated by fitting a two-component mixture over separation and centre, NOT by
splitting at +-45 deg. It is reported as NaN unless C4 clears the null, because
a separation angle is meaningless when there are no families to separate.
"""

import numpy as np


def effective_n(w):
    w = np.asarray(w, float)
    s = w.sum()
    return float(s * s / np.sum(w * w)) if s > 0 else 0.0


def _raw_moment(theta, w, k):
    w = np.asarray(w, float)
    return complex(np.sum(w * np.exp(1j * k * np.asarray(theta, float))) / w.sum())


def order_k(theta, w, k):
    """Null-corrected |<exp(ik.theta)>|. Zero for isotropic at any n."""
    if len(theta) == 0:
        return np.nan
    n = effective_n(w)
    if n < 2:
        return np.nan
    m = abs(_raw_moment(theta, w, k))
    corrected = (n * m * m - 1.0) / (n - 1.0)
    return float(np.sqrt(max(corrected, 0.0)))


def null_threshold(n, k=4, p=0.95):
    """Value of the RAW order parameter exceeded with probability 1-p under
    isotropy. Rayleigh: |m|^2 ~ Exp(1/n), so the p-quantile is -ln(1-p)/n."""
    if n < 2:
        return np.nan
    return float(np.sqrt(-np.log(1.0 - p) / n))


def _two_family_nll(theta, w, phi, delta, kappa):
    """Negative log-likelihood of a symmetric two-component von Mises mixture
    in doubled angles, with components at phi -+ delta/2."""
    a, b = phi - 0.5 * delta, phi + 0.5 * delta
    d1 = np.cos(2 * (theta - a))
    d2 = np.cos(2 * (theta - b))
    m = np.maximum(d1, d2) * kappa
    dens = np.exp(kappa * d1 - m) + np.exp(kappa * d2 - m)
    return -float(np.sum(w * (np.log(dens + 1e-300) + m)))


def interfamily_angle(theta, w, kappa=6.0, n_phi=90, n_delta=90):
    """Separation between the two fibre families, in degrees.

    Grid search over centre and separation. Does NOT assume 90 deg -- the
    search runs over 0..90 and can return any value in that range.
    """
    theta = np.asarray(theta, float)
    w = np.asarray(w, float)
    if len(theta) < 4:
        return np.nan
    phis = np.linspace(0, np.pi, n_phi, endpoint=False)
    deltas = np.linspace(np.radians(5.0), np.pi / 2, n_delta)
    best, best_d = np.inf, np.nan
    for d in deltas:
        for p in phis:
            v = _two_family_nll(theta, w, p, d, kappa)
            if v < best:
                best, best_d = v, d
    return float(np.degrees(best_d))


def analyse(theta, w=None, p_null=0.95):
    """Full orientational summary of a fibre population.

    Returns C2 (nematic), C4 (tetratic), both null-corrected, the director,
    the tetratic axis, whether C4 clears the isotropic null, and the
    interfamily angle (NaN when it does not).
    """
    theta = np.asarray(theta, float)
    if len(theta) == 0:
        return dict(n=0, n_eff=0.0, C2=np.nan, C4=np.nan, director_deg=np.nan,
                    tetratic_axis_deg=np.nan, C4_significant=False,
                    interfamily_deg=np.nan, C4_raw=np.nan, C4_null=np.nan)
    w = np.ones_like(theta) if w is None else np.asarray(w, float)
    n_eff = effective_n(w)
    m2, m4 = _raw_moment(theta, w, 2), _raw_moment(theta, w, 4)
    C2, C4 = order_k(theta, w, 2), order_k(theta, w, 4)
    thr = null_threshold(n_eff, 4, p_null)
    sig = bool(abs(m4) > thr)
    return dict(
        n=len(theta), n_eff=n_eff, C2=C2, C4=C4,
        C2_raw=abs(m2), C4_raw=abs(m4), C4_null=thr, C4_significant=sig,
        director_deg=float(np.degrees(0.5 * np.angle(m2) % np.pi)),
        tetratic_axis_deg=float(np.degrees(0.25 * np.angle(m4) % (np.pi / 2))),
        interfamily_deg=interfamily_angle(theta, w) if sig else np.nan)
