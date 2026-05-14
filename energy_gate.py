from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Utilities
def _safe_std(x: np.ndarray, eps: float = 1e-12) -> float:
    s = float(np.std(x))
    return s if s > eps else eps


def compute_channel_sigmas_from_runs(
    runs: List[Dict],
    start_t: Optional[float] = None,
    end_t: Optional[float] = None,
) -> Tuple[float, float, float]:
   
    rid_all, riq_all, rom_all = [], [], []
    for run in runs:
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


def residual_energy(
    r_id: float,
    r_iq: float,
    r_omega: float,
    sigmas: Tuple[float, float, float],
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Normalized squared energy:
        E = sum_i w_i * (r_i / sigma_i)^2
    """
    s_id, s_iq, s_om = sigmas
    w_id, w_iq, w_om = weights

    z_id = r_id / s_id
    z_iq = r_iq / s_iq
    z_om = r_omega / s_om

    return float(w_id * z_id * z_id + w_iq * z_iq * z_iq + w_om * z_om * z_om)


def ewma_filter(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple EWMA: y[k] = alpha*x[k] + (1-alpha)*y[k-1]
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if len(x) == 0:
        return y
    for k in range(1, len(x)):
        y[k] = alpha * x[k] + (1.0 - alpha) * y[k - 1]
    return y


def estimate_dt(t: np.ndarray) -> float:
    t = np.asarray(t)
    if len(t) < 2:
        return 1e-5
    return float(t[1] - t[0])

# Calibration

@dataclass(frozen=True)
class GateCalibration:
    sigmas: Tuple[float, float, float]
    T_high: float
    T_low: float
    alpha: float
    weights: Tuple[float, float, float]
    far: float
    meta: Dict


def calibrate_gate_from_healthy(
    healthy_runs: List[Dict],
    *,
    far: float = 0.01,
    alpha: float = 0.02,
    hysteresis_ratio: float = 0.6,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    # stats window for sigma estimation (optional)
    sigma_start_t: Optional[float] = None,
    sigma_end_t: Optional[float] = None,
    # energy window for threshold estimation (optional)
    energy_start_t: Optional[float] = None,
    energy_end_t: Optional[float] = None,
    # optional: exclude an initial transient from calibration
    drop_initial_seconds: float = 0.0,
) -> GateCalibration:

    if not healthy_runs:
        raise ValueError("healthy_runs is empty. Provide at least 1 healthy run.")

    # 1) channel scales
    sigmas = compute_channel_sigmas_from_runs(
        healthy_runs, start_t=sigma_start_t, end_t=sigma_end_t
    )

    # 2) build pooled EWMA energies
    pooled = []
    for run in healthy_runs:
        t = np.asarray(run["t"])
        rid = np.asarray(run["r_id"])
        riq = np.asarray(run["r_iq"])
        rom = np.asarray(run["r_omega"])

        mask = np.ones_like(t, dtype=bool)
        if drop_initial_seconds > 0:
            mask &= (t >= (t[0] + float(drop_initial_seconds)))
        if energy_start_t is not None:
            mask &= (t >= float(energy_start_t))
        if energy_end_t is not None:
            mask &= (t <= float(energy_end_t))

        # energy time-series
        E = np.array(
            [
                residual_energy(rid[k], riq[k], rom[k], sigmas, weights=weights)
                for k in range(len(t))
            ],
            dtype=float,
        )
        E = E[mask]
        if len(E) < 5:
            continue

        E_bar = ewma_filter(E, alpha=alpha)
        pooled.append(E_bar)

    if not pooled:
        raise ValueError(
            "Calibration failed: no valid samples pooled. "
            "Try reducing windowing constraints or drop_initial_seconds."
        )

    pooled_E = np.concatenate(pooled)

    # 3) quantile threshold
    far = float(far)
    far = min(max(far, 1e-6), 0.5)  # clamp
    q = 1.0 - far
    T_high = float(np.quantile(pooled_E, q))
    T_low = float(hysteresis_ratio * T_high)

    meta = {
        "n_runs": len(healthy_runs),
        "n_samples": int(len(pooled_E)),
        "quantile": q,
        "hysteresis_ratio": hysteresis_ratio,
        "drop_initial_seconds": drop_initial_seconds,
        "sigma_window": (sigma_start_t, sigma_end_t),
        "energy_window": (energy_start_t, energy_end_t),
    }

    return GateCalibration(
        sigmas=sigmas,
        T_high=T_high,
        T_low=T_low,
        alpha=float(alpha),
        weights=weights,
        far=far,
        meta=meta,
    )

# Gate State Machine

class ResidualEnergyGate:
    """
    Two-state gate with:
    - EWMA energy
    - hysteresis (T_high for ON, T_low for OFF)
    - persistence counters (N_on / N_off)
    - optional transient blanking based on omega_ref steps
    """

    def __init__(
        self,
        *,
        sigmas: Tuple[float, float, float],
        T_high: float,
        T_low: float,
        alpha: float = 0.02,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        N_on: int = 25,
        N_off: int = 25,
        # transient blanking:
        blanking_s: float = 0.05,
        domega_ref_threshold: float = 1e6,  # rad/s^2; set to np.inf to disable
        dt: Optional[float] = None,
    ):
        self.sigmas = sigmas
        self.T_high = float(T_high)
        self.T_low = float(T_low)
        self.alpha = float(alpha)
        self.weights = weights

        self.N_on = int(N_on)
        self.N_off = int(N_off)

        self.dt = dt  # if None, can be passed via update() call
        self.blanking_s = float(blanking_s)
        self.domega_ref_threshold = float(domega_ref_threshold)

        # internal state
        self.state = 0  # 0=HEALTHY (gate closed), 1=FAULT (gate open)
        self._counter = 0
        self._E_bar = 0.0
        self._blank_countdown = 0
        self._prev_omega_ref = None

    @property
    def energy_ewma(self) -> float:
        return float(self._E_bar)

    @property
    def blanking_active(self) -> bool:
        return self._blank_countdown > 0

    def reset(self):
        self.state = 0
        self._counter = 0
        self._E_bar = 0.0
        self._blank_countdown = 0
        self._prev_omega_ref = None

    def _update_blanking(self, omega_ref: Optional[float], dt: float):
        if omega_ref is None or not np.isfinite(self.domega_ref_threshold):
            return

        if self._prev_omega_ref is None:
            self._prev_omega_ref = float(omega_ref)
            return

        domega = (float(omega_ref) - float(self._prev_omega_ref)) / max(dt, 1e-12)
        self._prev_omega_ref = float(omega_ref)

        if abs(domega) > self.domega_ref_threshold:
            self._blank_countdown = int(round(self.blanking_s / max(dt, 1e-12)))

        if self._blank_countdown > 0:
            self._blank_countdown -= 1

    def update(
        self,
        r_id: float,
        r_iq: float,
        r_omega: float,
        *,
        omega_ref: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> int:
        """
        Returns
        -------
        state : int
            0 => HEALTHY (gate closed)
            1 => FAULT (gate open)
        """
        dt_eff = float(dt) if dt is not None else (float(self.dt) if self.dt is not None else 1e-5)

        # transient blanking logic
        self._update_blanking(omega_ref, dt_eff)
        if self.blanking_active:
            # force closed during blanking
            self.state = 0
            self._counter = 0
            return self.state

        # compute energy + EWMA
        E = residual_energy(r_id, r_iq, r_omega, self.sigmas, weights=self.weights)
        self._E_bar = self.alpha * E + (1.0 - self.alpha) * self._E_bar

        # hysteresis + persistence
        if self.state == 0:
            if self._E_bar > self.T_high:
                self._counter += 1
                if self._counter >= self.N_on:
                    self.state = 1
                    self._counter = 0
            else:
                self._counter = 0
        else:
            if self._E_bar < self.T_low:
                self._counter += 1
                if self._counter >= self.N_off:
                    self.state = 0
                    self._counter = 0
            else:
                self._counter = 0

        return self.state

# Batch helpers (evaluate a run)

def gate_run(
    run: Dict,
    gate: ResidualEnergyGate,
    *,
    start_t: Optional[float] = None,
    end_t: Optional[float] = None,
) -> Dict[str, np.ndarray]:
  
    t = np.asarray(run["t"])
    rid = np.asarray(run["r_id"])
    riq = np.asarray(run["r_iq"])
    rom = np.asarray(run["r_omega"])
    omega_ref = np.asarray(run["omega_ref"]) if "omega_ref" in run else None

    dt = estimate_dt(t)

    mask = np.ones_like(t, dtype=bool)
    if start_t is not None:
        mask &= (t >= float(start_t))
    if end_t is not None:
        mask &= (t <= float(end_t))

    idxs = np.where(mask)[0]
    states = np.zeros_like(t, dtype=int)
    ebar = np.zeros_like(t, dtype=float)

    gate.reset()
    for k in idxs:
        st = gate.update(
            float(rid[k]), float(riq[k]), float(rom[k]),
            omega_ref=(float(omega_ref[k]) if omega_ref is not None else None),
            dt=dt
        )
        states[k] = st
        ebar[k] = gate.energy_ewma

    return {"t": t, "state": states, "E_bar": ebar}

# Optional quick plotting 

def plot_gate_overlay(run: Dict, gate_out: Dict, title: str = "Residual Energy Gate"):
    
    import matplotlib.pyplot as plt

    t = gate_out["t"]
    E_bar = gate_out["E_bar"]
    state = gate_out["state"]

    plt.figure()
    plt.plot(t, E_bar, label="E_bar")
    plt.plot(t, state * (np.max(E_bar) * 0.9), label="gate_state (scaled)")
    plt.title(title)
    plt.xlabel("t [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()