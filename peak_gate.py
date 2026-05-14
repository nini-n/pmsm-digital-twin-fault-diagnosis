# utils/peak_gate.py
# Phase 3.2 — Transient Peak Detector (Ld mismatch gibi kısa/transient imzalar için)
#
# Mantık:
# - Healthy run'lardan normalized residual energy'nin, fault-on sonrası kısa penceredeki
#   "peak" dağılımını çıkar
# - Quantile ile T_peak threshold belirle
# - Test run'da aynı pencerede peak > T_peak ise fault_present=1

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


def _safe_std(x: np.ndarray, eps: float = 1e-12) -> float:
    s = float(np.std(x))
    return s if s > eps else eps


def estimate_dt(t: np.ndarray) -> float:
    t = np.asarray(t)
    if len(t) < 2:
        return 1e-5
    return float(t[1] - t[0])


def ewma_filter(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if len(x) == 0:
        return y
    for k in range(1, len(x)):
        y[k] = alpha * x[k] + (1.0 - alpha) * y[k - 1]
    return y


def compute_sigmas_from_healthy(
    healthy_runs: List[Dict],
    start_t: Optional[float] = None,
    end_t: Optional[float] = None,
) -> Tuple[float, float, float]:
    rid_all, riq_all, rom_all = [], [], []
    for run in healthy_runs:
        t = np.asarray(run["t"])
        rid = np.asarray(run["r_id"])
        riq = np.asarray(run["r_iq"])
        rom = np.asarray(run["r_omega"])

        mask = np.ones_like(t, dtype=bool)
        if start_t is not None:
            mask &= (t >= float(start_t))
        if end_t is not None:
            mask &= (t <= float(end_t))

        rid_all.append(rid[mask])
        riq_all.append(riq[mask])
        rom_all.append(rom[mask])

    rid_cat = np.concatenate(rid_all) if rid_all else np.array([0.0])
    riq_cat = np.concatenate(riq_all) if riq_all else np.array([0.0])
    rom_cat = np.concatenate(rom_all) if rom_all else np.array([0.0])

    return (_safe_std(rid_cat), _safe_std(riq_cat), _safe_std(rom_cat))


def residual_energy_series(
    run: Dict,
    sigmas: Tuple[float, float, float],
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.0),  # currents-only default
) -> np.ndarray:
    rid = np.asarray(run["r_id"])
    riq = np.asarray(run["r_iq"])
    rom = np.asarray(run["r_omega"])

    s_id, s_iq, s_om = sigmas
    w_id, w_iq, w_om = weights

    z_id = rid / s_id
    z_iq = riq / s_iq
    z_om = rom / s_om

    return (w_id * z_id * z_id + w_iq * z_iq * z_iq + w_om * z_om * z_om).astype(float)


@dataclass(frozen=True)
class PeakGateCalibration:
    sigmas: Tuple[float, float, float]
    T_peak: float
    alpha: float
    weights: Tuple[float, float, float]
    far: float
    win_s: float
    meta: Dict


def calibrate_peak_gate_from_healthy(
    healthy_runs: List[Dict],
    *,
    t_on: float = 0.10,
    win_s: float = 0.02,
    far: float = 0.01,
    alpha: float = 0.10,
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.0),
    sigma_start_t: Optional[float] = None,
    sigma_end_t: Optional[float] = None,
) -> PeakGateCalibration:
    if not healthy_runs:
        raise ValueError("healthy_runs is empty.")

    sigmas = compute_sigmas_from_healthy(
        healthy_runs, start_t=sigma_start_t, end_t=sigma_end_t
    )

    peaks = []
    for run in healthy_runs:
        t = np.asarray(run["t"])
        E = residual_energy_series(run, sigmas, weights=weights)
        E_bar = ewma_filter(E, alpha=alpha)

        mask = (t >= float(t_on)) & (t <= float(t_on + win_s))
        if np.any(mask):
            peaks.append(float(np.max(E_bar[mask])))

    if not peaks:
        raise ValueError("No peak samples found in window. Check t_on/win_s.")

    far = float(np.clip(far, 1e-6, 0.5))
    T_peak = float(np.quantile(np.array(peaks), 1.0 - far))

    meta = {
        "n_runs": len(healthy_runs),
        "n_peaks": len(peaks),
        "t_on": t_on,
        "win_s": win_s,
        "quantile": 1.0 - far,
    }

    return PeakGateCalibration(
        sigmas=sigmas,
        T_peak=T_peak,
        alpha=float(alpha),
        weights=weights,
        far=far,
        win_s=float(win_s),
        meta=meta,
    )


def peak_gate_decision(
    run: Dict,
    cal: PeakGateCalibration,
    *,
    t_on: float = 0.10,
) -> Tuple[int, float]:
    """
    Returns (decision, peak_value)
      decision: 1 => fault present, 0 => no-fault
    """
    t = np.asarray(run["t"])
    E = residual_energy_series(run, cal.sigmas, weights=cal.weights)
    E_bar = ewma_filter(E, alpha=cal.alpha)

    mask = (t >= float(t_on)) & (t <= float(t_on + cal.win_s))
    if not np.any(mask):
        return 0, 0.0

    peak = float(np.max(E_bar[mask]))
    decision = 1 if peak > cal.T_peak else 0
    return decision, peak