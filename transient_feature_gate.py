# utils/transient_feature_gate.py
# Phase 3.2 — Transient Feature Gate (Ld mismatch detection)
#
# Uses an energy-based log-score on (r_id, r_iq) in a short window after t_on:
#   score = log(sum(r_id^2)+eps) + log(sum(r_iq^2)+eps)
#
# Calibrates threshold from healthy distribution via quantile (1 - FAR).

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


def _energy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def compute_transient_score(run: Dict, t_on: float = 0.10, win_s: float = 0.02, eps: float = 1e-12) -> float:
    """
    score = log(E_id + eps) + log(E_iq + eps) over transient window [t_on, t_on + win_s]
    """
    t = np.asarray(run["t"])
    r_id = np.asarray(run["r_id"])
    r_iq = np.asarray(run["r_iq"])

    mask = (t >= float(t_on)) & (t <= float(t_on + win_s))
    if not np.any(mask):
        return 0.0

    E_id = _energy(r_id[mask])
    E_iq = _energy(r_iq[mask])

    return float(np.log(E_id + eps) + np.log(E_iq + eps))


@dataclass(frozen=True)
class TransientGateCalibration:
    T_tr: float
    far: float
    t_on: float
    win_s: float
    meta: Dict


def calibrate_transient_gate_from_healthy(
    healthy_runs: List[Dict],
    *,
    t_on: float = 0.10,
    win_s: float = 0.02,
    far: float = 0.01
) -> TransientGateCalibration:
    vals = []
    for run in healthy_runs:
        vals.append(compute_transient_score(run, t_on=t_on, win_s=win_s))

    if not vals:
        raise ValueError("No transient scores computed.")

    vals = np.array(vals, dtype=float)
    far = float(np.clip(far, 1e-6, 0.5))
    T_tr = float(np.quantile(vals, 1.0 - far))

    meta = {
        "n_runs": int(len(vals)),
        "quantile": 1.0 - far,
        "mean_healthy": float(np.mean(vals)),
        "std_healthy": float(np.std(vals)),
        "min_healthy": float(np.min(vals)),
        "max_healthy": float(np.max(vals)),
    }

    return TransientGateCalibration(T_tr=T_tr, far=far, t_on=t_on, win_s=win_s, meta=meta)


def transient_gate_decision(run: Dict, cal: TransientGateCalibration):
    val = compute_transient_score(run, t_on=cal.t_on, win_s=cal.win_s)
    decision = 1 if val > cal.T_tr else 0
    return decision, val