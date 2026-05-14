# main.py (Phase 3.1 - Parallel Nominal Twin Residual Generator)

from models.pmsm import PMSM
from controllers.pi import PIController
import numpy as np
import matplotlib.pyplot as plt
import os, json, csv
from datetime import datetime


# -----------------------------
# Fault engine
# -----------------------------
def apply_faults(motor, t, fault_cfg=None):
    """
    Reset motor params to nominal every step, then apply scheduled fault after t_on.

    fault_cfg example:
    {"type": "rs_drift", "t_on": 0.10, "severity": 0.20}
    {"type": "ld_mismatch", "t_on": 0.10, "severity": -0.10}
    """
    motor.Rs = motor.Rs_nominal
    motor.Ld = motor.Ld_nominal

    if fault_cfg is None:
        return "healthy", 0.0

    ftype = fault_cfg.get("type", "healthy")
    t_on = float(fault_cfg.get("t_on", 1e9))
    sev = float(fault_cfg.get("severity", 0.0))

    if t >= t_on:
        if ftype == "rs_drift":
            motor.Rs = motor.Rs_nominal * (1.0 + sev)
            return "rs_drift", sev

        if ftype == "ld_mismatch":
            motor.Ld = motor.Ld_nominal * (1.0 + sev)
            return "ld_mismatch", sev

    return "healthy", 0.0


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_run_to_csv(run_dict, out_csv_path):
    keys = [
        "t",
        "omega", "omega_ref",
        "id", "iq", "iq_ref",
        "Te", "Tl",
        "vd", "vq",
        "vd_pi", "vq_pi",
        "vd_ff", "vq_ff",
        # Phase 3.1 additions
        "id_hat", "iq_hat", "omega_hat",
        "r_id", "r_iq", "r_omega",
    ]
    n = len(run_dict["t"])
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([run_dict[k][i] for k in keys])


def sev_to_tag(sev):
    """
    Convert severity float to compact tag.
    +0.10 -> p10
    -0.20 -> m20
    0.05  -> p05
    """
    s = int(round(sev * 100))
    if s >= 0:
        return f"p{abs(s):02d}"
    return f"m{abs(s):02d}"


# -----------------------------
# Feature extraction (Phase 3: residuals are inside the run)
# -----------------------------
def compute_features_from_residual(run, t_on=0.10):
    """
    Features computed from Phase-3 residual signals:
      r_id = id_meas - id_hat
      r_iq = iq_meas - iq_hat
      r_omega = omega_meas - omega_hat
    """
    t = run["t"]
    dt_est = float(t[1] - t[0]) if len(t) > 1 else 1e-5

    r_id = run["r_id"]
    r_iq = run["r_iq"]
    r_om = run["r_omega"]

    # windows
    tr_mask = (t >= t_on) & (t < t_on + 0.02)   # 20 ms transient
    ss_mask = (t >= t_on + 0.05)                # steady after 50 ms

    def safe(x, mask):
        xx = x[mask]
        return xx if len(xx) else np.array([0.0])

    def rms(x, mask):
        xx = safe(x, mask)
        return float(np.sqrt(np.mean(xx * xx)))

    def maxabs(x, mask):
        xx = safe(x, mask)
        return float(np.max(np.abs(xx)))

    def mean(x, mask):
        xx = safe(x, mask)
        return float(np.mean(xx))

    def std(x, mask):
        xx = safe(x, mask)
        return float(np.std(xx))

    def energy(x, mask):
        xx = safe(x, mask)
        return float(np.sum(xx * xx))

    # Frequency-domain: dominant frequency + dominance ratio
    def dom_freq_and_ratio(x, mask, dt):
        xx = safe(x, mask)
        if len(xx) < 16:
            return 0.0, 0.0
        xx = xx - np.mean(xx)  # remove DC
        w = np.hanning(len(xx))
        X = np.fft.rfft(xx * w)
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(len(xx), d=dt)
        if len(mag) < 2:
            return 0.0, 0.0
        mag[0] = 0.0  # ignore DC bin
        kmax = int(np.argmax(mag))
        f_dom = float(freqs[kmax])
        ratio = float(mag[kmax] / (np.sum(mag) + 1e-12))
        return f_dom, ratio

    f_iq, fr_iq = dom_freq_and_ratio(r_iq, tr_mask, dt_est)
    f_id, fr_id = dom_freq_and_ratio(r_id, tr_mask, dt_est)

    return {
        # steady: consistency drift
        "rms_r_iq_ss": rms(r_iq, ss_mask),
        "maxabs_r_iq_ss": maxabs(r_iq, ss_mask),
        "mean_r_iq_ss": mean(r_iq, ss_mask),
        "std_r_iq_ss": std(r_iq, ss_mask),

        "rms_r_id_ss": rms(r_id, ss_mask),
        "maxabs_r_id_ss": maxabs(r_id, ss_mask),
        "mean_r_id_ss": mean(r_id, ss_mask),
        "std_r_id_ss": std(r_id, ss_mask),

        "rms_r_om_ss": rms(r_om, ss_mask),
        "maxabs_r_om_ss": maxabs(r_om, ss_mask),

        # transient: mismatch shows here
        "rms_r_iq_tr": rms(r_iq, tr_mask),
        "maxabs_r_iq_tr": maxabs(r_iq, tr_mask),
        "energy_r_iq_tr": energy(r_iq, tr_mask),
        "fdom_r_iq_tr": f_iq,
        "fratio_r_iq_tr": fr_iq,

        "rms_r_id_tr": rms(r_id, tr_mask),
        "maxabs_r_id_tr": maxabs(r_id, tr_mask),
        "energy_r_id_tr": energy(r_id, tr_mask),
        "fdom_r_id_tr": f_id,
        "fratio_r_id_tr": fr_id,
    }


# -----------------------------
# Core simulation (Drive + Fault + Scenario + Noise) - Phase 3.1
# -----------------------------
def run_simulation(params, fault_cfg=None, scenario=None, seed=0, noise_std=0.0):
    # Faulty plant + nominal twin
    motor_f = PMSM(params)  # faulty plant (faults applied)
    motor_n = PMSM(params)  # nominal twin (no faults)

    np.random.seed(int(seed))

    if scenario is None:
        scenario = {
            "omega_step": 50.0,
            "t_speed": 0.02,
            "Tl0": 0.5,
            "Tl1": 1.0,
            "t_load": 0.12,
        }

    dt = 1e-5
    Tsim = 0.2
    steps = int(Tsim / dt)

    # Inner current loop tuning (bandwidth method)
    bw_i = 500.0
    Kp_d = params["Ld"] * bw_i
    Ki_d = params["Rs"] * bw_i
    Kp_q = params["Lq"] * bw_i
    Ki_q = params["Rs"] * bw_i

    v_limit = 200.0
    pi_d = PIController(Kp_d, Ki_d, limit=v_limit)
    pi_q = PIController(Kp_q, Ki_q, limit=v_limit)

    # Outer speed loop tuning
    bw_w = 25.0
    Kt = 1.5 * params["p"] * params["lambda_m"]
    Kp_w = (params["J"] * bw_w) / Kt
    Ki_w = 0.5 * Kp_w * bw_w

    iq_limit = 30.0
    pi_w = PIController(Kp_w, Ki_w, limit=iq_limit)

    id_ref = 0.0

    # logs
    t_log = np.zeros(steps)

    omega_log = np.zeros(steps)
    omega_ref_log = np.zeros(steps)

    id_log = np.zeros(steps)
    iq_log = np.zeros(steps)
    iq_ref_log = np.zeros(steps)

    Te_log = np.zeros(steps)
    Tl_log = np.zeros(steps)

    vd_log = np.zeros(steps)
    vq_log = np.zeros(steps)
    vd_pi_log = np.zeros(steps)
    vq_pi_log = np.zeros(steps)
    vd_ff_log = np.zeros(steps)
    vq_ff_log = np.zeros(steps)

    fault_label_log = np.empty(steps, dtype=object)
    fault_sev_log = np.zeros(steps)

    # twin states
    id_hat_log = np.zeros(steps)
    iq_hat_log = np.zeros(steps)
    omega_hat_log = np.zeros(steps)

    # residual logs
    r_id_log = np.zeros(steps)
    r_iq_log = np.zeros(steps)
    r_omega_log = np.zeros(steps)

    for k in range(steps):
        t = k * dt

        omega_ref = 0.0 if t < float(scenario["t_speed"]) else float(scenario["omega_step"])
        Tl = float(scenario["Tl0"]) if t < float(scenario["t_load"]) else float(scenario["Tl1"])

        # Apply faults ONLY to faulty plant
        flabel, fsev = apply_faults(motor_f, t, fault_cfg)

        # Speed loop uses faulty plant speed
        err_w = omega_ref - motor_f.omega
        iq_ref = pi_w.update(err_w, dt)

        # Current loop uses faulty plant currents
        err_d = id_ref - motor_f.id
        err_q = iq_ref - motor_f.iq
        vd_pi = pi_d.update(err_d, dt)
        vq_pi = pi_q.update(err_q, dt)

        # Decoupling feedforward (controller uses nominal params)
        omega_e = params["p"] * motor_f.omega
        vd_ff = -omega_e * params["Lq"] * motor_f.iq
        vq_ff = (omega_e * params["Ld"] * motor_f.id) + (omega_e * params["lambda_m"])

        vd = vd_pi + vd_ff
        vq = vq_pi + vq_ff

        # Clamp
        vd = max(-v_limit, min(v_limit, vd))
        vq = max(-v_limit, min(v_limit, vq))

        # Step both systems with SAME inputs
        id_f, iq_f, omega_f, Te_f = motor_f.step(vd=vd, vq=vq, Tl=Tl, dt=dt)
        id_n, iq_n, omega_n, Te_n = motor_n.step(vd=vd, vq=vq, Tl=Tl, dt=dt)

        # Measurement noise on faulty measurements
        if noise_std and noise_std > 0.0:
            omega_meas = omega_f + np.random.normal(0.0, noise_std)
            iq_meas = iq_f + np.random.normal(0.0, noise_std)
            id_meas = id_f + np.random.normal(0.0, noise_std)
        else:
            omega_meas = omega_f
            iq_meas = iq_f
            id_meas = id_f

        # Residuals (measured - predicted)
        r_id = id_meas - id_n
        r_iq = iq_meas - iq_n
        r_omega = omega_meas - omega_n

        # Log
        t_log[k] = t
        omega_log[k] = omega_meas
        omega_ref_log[k] = omega_ref

        id_log[k] = id_meas
        iq_log[k] = iq_meas
        iq_ref_log[k] = iq_ref

        Te_log[k] = Te_f
        Tl_log[k] = Tl

        vd_log[k] = vd
        vq_log[k] = vq
        vd_pi_log[k] = vd_pi
        vq_pi_log[k] = vq_pi
        vd_ff_log[k] = vd_ff
        vq_ff_log[k] = vq_ff

        fault_label_log[k] = flabel
        fault_sev_log[k] = fsev

        id_hat_log[k] = id_n
        iq_hat_log[k] = iq_n
        omega_hat_log[k] = omega_n

        r_id_log[k] = r_id
        r_iq_log[k] = r_iq
        r_omega_log[k] = r_omega

    return {
        "t": t_log,
        "omega": omega_log,
        "omega_ref": omega_ref_log,
        "id": id_log,
        "iq": iq_log,
        "iq_ref": iq_ref_log,
        "Te": Te_log,
        "Tl": Tl_log,
        "vd": vd_log,
        "vq": vq_log,
        "vd_pi": vd_pi_log,
        "vq_pi": vq_pi_log,
        "vd_ff": vd_ff_log,
        "vq_ff": vq_ff_log,
        "fault_label": fault_label_log,
        "fault_severity": fault_sev_log,

        # Phase 3.1 additions
        "id_hat": id_hat_log,
        "iq_hat": iq_hat_log,
        "omega_hat": omega_hat_log,
        "r_id": r_id_log,
        "r_iq": r_iq_log,
        "r_omega": r_omega_log,

        "meta": {
            "dt": dt,
            "Tsim": Tsim,
            "bw_i": bw_i,
            "bw_w": bw_w,
            "Kp_w": Kp_w,
            "Ki_w": Ki_w,
            "Kp_d": Kp_d,
            "Ki_d": Ki_d,
            "Kp_q": Kp_q,
            "Ki_q": Ki_q,
            "fault_cfg": fault_cfg,
            "phase": "phase3_parallel_twin",
        },
    }


# -----------------------------
# Dataset generation (Multi-fault factorial design) + Optional ML
# -----------------------------
def main():
    params = {
        "Rs": 0.5,
        "Ld": 0.001,
        "Lq": 0.001,
        "lambda_m": 0.05,
        "p": 4,
        "J": 0.001,
        "B": 0.0001
    }

    ensure_dir("data/runs")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    omega_steps = [40.0, 50.0, 60.0]
    load_profiles = [
        {"Tl0": 0.5, "Tl1": 1.0, "t_load": 0.12},
        {"Tl0": 0.3, "Tl1": 0.8, "t_load": 0.10},
        {"Tl0": 0.6, "Tl1": 1.2, "t_load": 0.15},
    ]
    seeds = [0, 1, 2]
    t_on = 0.10
    noise_std = 0.01

    # Healthy replicates per scenario+seed (balances classes)
    healthy_reps = 3

    # multi-fault plan
    fault_plans = [
        ("rs_drift", [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]),
        ("ld_mismatch", [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]),
    ]

    index_rows = []
    feature_rows = []

    for omega_step in omega_steps:
        for lp_idx, lp in enumerate(load_profiles):
            for seed in seeds:
                scenario = {
                    "omega_step": omega_step,
                    "t_speed": 0.02,
                    "Tl0": lp["Tl0"],
                    "Tl1": lp["Tl1"],
                    "t_load": lp["t_load"],
                }

                base_tag = f"{run_id}_w{int(omega_step)}_lp{lp_idx}_seed{seed}"

                # -------- healthy replicates --------
                for rep in range(healthy_reps):
                    rep_seed = seed + 1000 * rep
                    healthy = run_simulation(
                        params,
                        fault_cfg=None,
                        scenario=scenario,
                        seed=rep_seed,
                        noise_std=noise_std
                    )

                    healthy_tag = f"{base_tag}_healthy_r{rep}"
                    healthy_dir = f"data/runs/{healthy_tag}"
                    ensure_dir(healthy_dir)
                    save_run_to_csv(healthy, f"{healthy_dir}/signals.csv")
                    with open(f"{healthy_dir}/meta.json", "w") as f:
                        json.dump(
                            {
                                "tag": healthy_tag,
                                "fault_type": "healthy",
                                "severity": 0.0,
                                "fault_cfg": None,
                                "scenario": scenario,
                                "seed": rep_seed,
                                "noise_std": noise_std,
                                **healthy["meta"],
                            },
                            f, indent=2
                        )

                    index_rows.append({
                        "tag": healthy_tag,
                        "signals_csv": f"{healthy_dir}/signals.csv",
                        "meta_json": f"{healthy_dir}/meta.json",
                        "fault_type": "healthy",
                        "severity": 0.0,
                        "omega_step": omega_step,
                        "load_profile": lp_idx,
                        "seed": seed,  # scenario grouping seed
                        "noise_std": noise_std,
                    })

                    feats_h = compute_features_from_residual(healthy, t_on=t_on)
                    feature_rows.append({
                        "tag": healthy_tag,
                        "fault_type": "healthy",
                        "severity": 0.0,
                        "omega_step": omega_step,
                        "load_profile": lp_idx,
                        "seed": seed,
                        **feats_h
                    })

                # -------- faults --------
                for fault_type, severities in fault_plans:
                    for sev in severities:
                        fault_cfg = {"type": fault_type, "t_on": t_on, "severity": sev}
                        run = run_simulation(
                            params,
                            fault_cfg=fault_cfg,
                            scenario=scenario,
                            seed=seed,
                            noise_std=noise_std
                        )

                        tag = f"{base_tag}_{fault_type}_{sev_to_tag(sev)}"
                        out_dir = f"data/runs/{tag}"
                        ensure_dir(out_dir)

                        save_run_to_csv(run, f"{out_dir}/signals.csv")
                        with open(f"{out_dir}/meta.json", "w") as f:
                            json.dump(
                                {
                                    "tag": tag,
                                    "fault_type": fault_type,
                                    "severity": sev,
                                    "fault_cfg": fault_cfg,
                                    "scenario": scenario,
                                    "seed": seed,
                                    "noise_std": noise_std,
                                    **run["meta"],
                                },
                                f, indent=2
                            )

                        index_rows.append({
                            "tag": tag,
                            "signals_csv": f"{out_dir}/signals.csv",
                            "meta_json": f"{out_dir}/meta.json",
                            "fault_type": fault_type,
                            "severity": sev,
                            "omega_step": omega_step,
                            "load_profile": lp_idx,
                            "seed": seed,
                            "noise_std": noise_std,
                        })

                        feats = compute_features_from_residual(run, t_on=t_on)
                        feature_rows.append({
                            "tag": tag,
                            "fault_type": fault_type,
                            "severity": sev,
                            "omega_step": omega_step,
                            "load_profile": lp_idx,
                            "seed": seed,
                            **feats
                        })

    # save index
    index_path = f"data/runs_index_{run_id}.csv"
    with open(index_path, "w", newline="") as f:
        cols = list(index_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in index_rows:
            w.writerow(r)

    # save features
    feat_path = f"data/features_{run_id}.csv"
    with open(feat_path, "w", newline="") as f:
        cols = list(feature_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in feature_rows:
            w.writerow(r)

    print(f"[OK] Dataset generated: {len(index_rows)} runs")
    print(f"[OK] Index: {index_path}")
    print(f"[OK] Features: {feat_path}")

    # -----------------------------
    # Optional: quick multi-class baseline (scenario-aware split)
    # -----------------------------
    try:
        import pandas as pd
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("\n(Optional ML) Missing packages. Install with:")
        print("python -m pip install pandas scikit-learn")
        return

    df = pd.read_csv(feat_path)

    feature_cols = [
        "rms_r_iq_ss", "maxabs_r_iq_ss", "mean_r_iq_ss", "std_r_iq_ss",
        "rms_r_id_ss", "maxabs_r_id_ss", "mean_r_id_ss", "std_r_id_ss",
        "rms_r_om_ss", "maxabs_r_om_ss",
        "rms_r_iq_tr", "maxabs_r_iq_tr", "energy_r_iq_tr", "fdom_r_iq_tr", "fratio_r_iq_tr",
        "rms_r_id_tr", "maxabs_r_id_tr", "energy_r_id_tr", "fdom_r_id_tr", "fratio_r_id_tr",
    ]

    X = df[feature_cols].values
    y = df["fault_type"].values

    groups = (
        df["omega_step"].astype(str)
        + "_" + df["load_profile"].astype(str)
        + "_" + df["seed"].astype(str)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])

    print("\n[Multi-class classification: healthy vs rs_drift vs ld_mismatch]")
    print(classification_report(y[test_idx], y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y[test_idx], y_pred))


if __name__ == "__main__":
    main()