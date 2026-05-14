# test_gate_phase32.py
import pandas as pd
import numpy as np

from utils.energy_gate import (
    calibrate_gate_from_healthy,
    ResidualEnergyGate,
    gate_run,
)

from utils.transient_feature_gate import (
    calibrate_transient_gate_from_healthy,
    transient_gate_decision,
)

RUNS_INDEX = "data/runs_index_20260301_204447.csv"


def load_run(signals_path: str):
    df = pd.read_csv(signals_path, usecols=["t", "omega_ref", "r_id", "r_iq", "r_omega"])
    return {
        "t": df["t"].to_numpy(),
        "omega_ref": df["omega_ref"].to_numpy(),
        "r_id": df["r_id"].to_numpy(),
        "r_iq": df["r_iq"].to_numpy(),
        "r_omega": df["r_omega"].to_numpy(),
    }


def summarize_gate(name: str, out: dict):
    state = out["state"]
    ebar = out["E_bar"]
    open_ratio = float(np.mean(state == 1))
    print(f"[{name}] gate open ratio:", open_ratio)
    print(f"[{name}] max E_bar:", float(np.max(ebar)))
    return open_ratio


def final_detection(out_slow: dict, tr_decision: int) -> int:
    return int((np.any(out_slow["state"] == 1)) or (tr_decision == 1))


def main():
    idx = pd.read_csv(RUNS_INDEX)

    # -----------------------------
    # 1) Healthy runs for calibration
    # -----------------------------
    healthy = idx[idx["fault_type"] == "healthy"].copy()
    healthy = healthy.sample(n=min(30, len(healthy)), random_state=42)
    healthy_runs = [load_run(p) for p in healthy["signals_csv"].tolist()]

    # -----------------------------
    # 2) Slow gate calibration (steady faults: Rs drift)
    # -----------------------------
    cal_slow = calibrate_gate_from_healthy(
        healthy_runs,
        far=0.005,
        alpha=0.02,
        hysteresis_ratio=0.6,
        drop_initial_seconds=0.02,
        weights=(1.0, 1.0, 1.0),
    )

    print("\n[CAL-SLOW] sigmas:", cal_slow.sigmas)
    print("[CAL-SLOW] T_high:", cal_slow.T_high, "T_low:", cal_slow.T_low)

    gate_slow = ResidualEnergyGate(
        sigmas=cal_slow.sigmas,
        T_high=cal_slow.T_high,
        T_low=cal_slow.T_low,
        alpha=cal_slow.alpha,
        weights=cal_slow.weights,
        N_on=50,
        N_off=50,
        blanking_s=0.05,
        domega_ref_threshold=1e6,
    )

    # -----------------------------
    # 3) Transient score gate calibration (Ld mismatch)
    # -----------------------------
    tr_cal = calibrate_transient_gate_from_healthy(
        healthy_runs,
        t_on=0.10,
        win_s=0.02,
        far=0.01
    )

    print("\n[TR-CAL] T_tr:", tr_cal.T_tr)
    print("[TR-CAL] meta:", tr_cal.meta)

    # -----------------------------
    # 4) Sanity test on ONE healthy
    # -----------------------------
    one_h_path = healthy["signals_csv"].iloc[0]
    print("\n[TEST] healthy using:", one_h_path)
    run_h = load_run(one_h_path)

    out_h_slow = gate_run(run_h, gate_slow)
    summarize_gate("HEALTHY-SLOW", out_h_slow)

    dec_tr_h, val_tr_h = transient_gate_decision(run_h, tr_cal)
    print("[HEALTHY-TR] decision:", dec_tr_h, "score:", val_tr_h)

    detect_h = final_detection(out_h_slow, dec_tr_h)
    print("[HEALTHY-FINAL-DETECTION] fault_present:", detect_h)

    # -----------------------------
    # 5) Fault tests
    # -----------------------------
    # rs_drift: one sample
    subset_rs = idx[idx["fault_type"] == "rs_drift"].copy()
    one_rs_path = subset_rs["signals_csv"].iloc[0]
    print("\n[FAULT TEST] rs_drift using:", one_rs_path)
    run_rs = load_run(one_rs_path)
    out_rs_slow = gate_run(run_rs, gate_slow)
    summarize_gate("RS_DRIFT-SLOW", out_rs_slow)
    dec_tr_rs, val_tr_rs = transient_gate_decision(run_rs, tr_cal)
    print("[RS_DRIFT-TR] decision:", dec_tr_rs, "score:", val_tr_rs)
    print("[RS_DRIFT-FINAL-DETECTION] fault_present:", final_detection(out_rs_slow, dec_tr_rs))

    # ld_mismatch: test multiple severities if available
    subset_ld = idx[idx["fault_type"] == "ld_mismatch"].copy()

    # try a few different severities (if present)
    desired = ["m20", "m10", "p10", "p20", "m05", "p05"]
    picked = []
    for tag in desired:
        rows = subset_ld[subset_ld["tag"].astype(str).str.contains(f"_{tag}", regex=False)]
        if len(rows) > 0:
            picked.append(rows.iloc[0]["signals_csv"])
        if len(picked) >= 3:
            break

    if not picked:
        picked = [subset_ld["signals_csv"].iloc[0]]

    for pth in picked:
        print("\n[FAULT TEST] ld_mismatch using:", pth)
        run_ld = load_run(pth)

        out_ld_slow = gate_run(run_ld, gate_slow)
        summarize_gate("LD_MISMATCH-SLOW", out_ld_slow)

        dec_tr_ld, val_tr_ld = transient_gate_decision(run_ld, tr_cal)
        print("[LD_MISMATCH-TR] decision:", dec_tr_ld, "score:", val_tr_ld)

        detect_ld = final_detection(out_ld_slow, dec_tr_ld)
        print("[LD_MISMATCH-FINAL-DETECTION] fault_present:", detect_ld)

    print("\n[OK] Phase 3.2 detection test finished (slow OR transient log-energy score).")


if __name__ == "__main__":
    main()