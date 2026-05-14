import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier

# =============================
# Thesis figure style
# =============================
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "petrol": "#0B3C49",
    "teal": "#1B998B",
    "copper": "#C46A2C",
    "plum": "#5D2E8C",
    "slate": "#334E68",
    "sand": "#D9A441",
    "grid": "#D6D2C4",
}

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Change this manually if you want a specific file.
FEATURES_PATH = None

if FEATURES_PATH is None:
    candidates = sorted(glob.glob("data/features_*.csv"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError("No data/features_*.csv file found.")
    FEATURES_PATH = candidates[-1]

print(f"[INFO] Loading: {FEATURES_PATH}")
df = pd.read_csv(FEATURES_PATH)

# Supports both earlier Rs-only feature files and later Phase-3 residual feature files.
old_feature_cols = [
    "rms_r_vq", "mean_r_vq", "std_r_vq", "maxabs_r_vq", "slope_r_vq",
    "rms_r_iq", "maxabs_r_iq", "rms_r_omega", "maxabs_r_omega"
]
phase3_feature_cols = [
    "rms_r_iq_ss", "maxabs_r_iq_ss", "mean_r_iq_ss", "std_r_iq_ss",
    "rms_r_id_ss", "maxabs_r_id_ss", "mean_r_id_ss", "std_r_id_ss",
    "rms_r_om_ss", "maxabs_r_om_ss",
    "rms_r_iq_tr", "maxabs_r_iq_tr", "energy_r_iq_tr", "fdom_r_iq_tr", "fratio_r_iq_tr",
    "rms_r_id_tr", "maxabs_r_id_tr", "energy_r_id_tr", "fdom_r_id_tr", "fratio_r_id_tr",
]

if all(c in df.columns for c in old_feature_cols):
    feature_cols = old_feature_cols
elif all(c in df.columns for c in phase3_feature_cols):
    feature_cols = phase3_feature_cols
else:
    raise ValueError("Feature columns in CSV do not match old or Phase-3 feature set.")

X = df[feature_cols].values
severity = df["severity"].values if "severity" in df.columns else np.zeros(len(df))

if "omega_step" in df.columns and "load_profile" in df.columns:
    groups = df["omega_step"].astype(str) + "_" + df["load_profile"].astype(str)
else:
    groups = np.arange(len(df)).astype(str)

# =============================
# Classification feature importance
# =============================
if "fault_type" in df.columns:
    y_cls = df["fault_type"].values
else:
    y_cls = (severity > 0).astype(int)

gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(gss.split(X, y_cls, groups=groups))
clf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
clf.fit(X[train_idx], y_cls[train_idx])
importance = clf.feature_importances_
order = np.argsort(importance)

fig, ax = plt.subplots(figsize=(8.2, 5.4))
colors = [COLORS["petrol"] if i % 2 == 0 else COLORS["teal"] for i in range(len(order))]
ax.barh(np.array(feature_cols)[order], importance[order], color=colors, edgecolor="#243B53", linewidth=0.5)
ax.set_title("Feature Contribution to Fault Classification", pad=14, fontweight="bold")
ax.set_xlabel("Relative importance")
ax.grid(axis="x", color=COLORS["grid"], linestyle="--", linewidth=0.8, alpha=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "thesis_feature_importance.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "thesis_feature_importance.svg"), bbox_inches="tight")
plt.show()

# =============================
# Regression calibration, if severity has enough distinct values
# =============================
if len(np.unique(severity)) > 2:
    y_reg = severity.astype(float)
    train_idx, test_idx = next(gss.split(X, y_reg, groups=groups))
    model = Ridge(alpha=1e-3)
    model.fit(X[train_idx], y_reg[train_idx])
    y_pred = model.predict(X[test_idx])

    mae = mean_absolute_error(y_reg[test_idx], y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg[test_idx], y_pred))
    r2 = r2_score(y_reg[test_idx], y_pred)
    print(f"[REG] MAE={mae:.5f}, RMSE={rmse:.5f}, R2={r2:.5f}")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(y_reg[test_idx], y_pred, s=44, color=COLORS["plum"], alpha=0.86,
               edgecolor="white", linewidth=0.6, label="test samples")
    lo = min(y_reg[test_idx].min(), y_pred.min())
    hi = max(y_reg[test_idx].max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=COLORS["copper"], linewidth=2.0,
            label="ideal calibration")
    ax.set_title("Severity Calibration", pad=14, fontweight="bold")
    ax.set_xlabel("True severity")
    ax.set_ylabel("Predicted severity")
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "thesis_severity_calibration.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "thesis_severity_calibration.svg"), bbox_inches="tight")
    plt.show()

    residual = y_pred - y_reg[test_idx]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.hist(residual, bins=22, color=COLORS["slate"], edgecolor="white", linewidth=0.7)
    ax.axvline(0, color=COLORS["copper"], linestyle="--", linewidth=1.8)
    ax.set_title("Severity Regression Residuals", pad=14, fontweight="bold")
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "thesis_regression_residuals.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "thesis_regression_residuals.svg"), bbox_inches="tight")
    plt.show()
