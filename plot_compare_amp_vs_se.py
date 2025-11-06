# plot_compare_amp_vs_se.py
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======= USER SETTINGS =======
# --- AMP side (point to the specific gamma/delta/plant directory) ---
# ======= USER SETTINGS (edit these by hand) =======

# --- Common experiment parameters ---
DATE = "20251030"              # folder date stamp used in AMP path
BETA_U = 1.0
BETA_V = 2.0
D = 1000
TOL = 1e-4
DAMP = 0.9
MAX_ITER = 500
EPS = 1e-6
RUNS = 72
ALPHA_MIN = 0.1
ALPHA_MAX = 80
N_ALPHA = 50

# --- Which (gamma, delta, plant) to plot ---
GAMMA = 0.0
DELTA = 0.0
PLANT = 0.9

# --- SE (state evolution) parameters ---
SE_MODE = "replica"   # or "rBP"
SE_QINIT = np.array([[0.90, 0.10],
                     [0.10, 0.90]])
SE_SAMPLES = 1000
SE_ITERS = 50
SE_DAMP = 0.3





# ===================================================

def alabel(a: float) -> str:
    """Format numbers safely for folder names: 0.5 → '0p5', 1e-4 → '1p00000e-04'."""
    s = f"{a:.6g}"
    return s.replace(".", "p").replace("-", "m")

def alabel3(x: float, places: int = 3) -> str:
    # 1.000 -> "1p000"; 0.010 -> "0p010"; 0.3 -> "0p300"
    s = f"{x:.{places}f}"
    return s.replace(".", "p").replace("-", "m")


# ======= AUTOMATIC PATH GENERATION =======

AMP_BASE = Path("amp_runs")
AMP_TOP = AMP_BASE / DATE / "_".join([
    f"betaU={alabel(BETA_U)}",
    f"betaV={alabel(BETA_V)}",
    f"d={D}",
    f"tol={alabel(TOL)}",
    f"damp={alabel(DAMP)}",
    f"maxit={MAX_ITER}",
    f"eps={alabel(EPS)}",
    f"runs={RUNS}",
    f"alphaMin={alabel(ALPHA_MIN)}",
    f"alphaMax={alabel(ALPHA_MAX)}",
    f"nAlpha={N_ALPHA}",
])
AMP_GD_DIR = AMP_TOP / f"plant={alabel(PLANT)}" / f"gamma={alabel(GAMMA)}_delta={alabel(DELTA)}"

SE_BASE = Path("runs")
SE_QTAG = f"qinit={alabel3(SE_QINIT[0,0])}_{alabel3(SE_QINIT[1,1])}_{alabel3(SE_QINIT[0,1])}"
SE_TOP = SE_BASE / "_".join([
    f"betaU={alabel3(BETA_U)}",
    f"betaV={alabel3(BETA_V)}",
    SE_QTAG,
    f"samples={SE_SAMPLES}",
    f"iters={SE_ITERS}",
    f"damp={alabel3(SE_DAMP)}",
])
SE_GD_DIR = SE_TOP / f"mode={SE_MODE}" / f"gamma={alabel3(GAMMA)}_delta={alabel3(DELTA)}"

CONVERGED_ONLY = True

# ======= OUTPUT FIGURE PATH (auto-generated) =======

# --- Plot metadata ---
TITLE = "Overlaps vs α — SE (solid) vs AMP (×)"
# Base folder for figures
FIG_BASE = Path("figures")

# Make sure folder exists
FIG_BASE.mkdir(parents=True, exist_ok=True)

# Build an informative filename that encodes key parameters
fig_tag_parts = [
    f"gamma={alabel(GAMMA)}",
    f"delta={alabel(DELTA)}",
    f"plant={alabel(PLANT)}",
    f"betaU={alabel(BETA_U)}",
    f"betaV={alabel(BETA_V)}",
    f"d={D}",
    SE_MODE
]

if CONVERGED_ONLY:
    fig_tag_parts.append("converged_only")

fig_tag = "_".join(fig_tag_parts)
                

# Final path (you can change prefix if you want)
SAVE_PNG = FIG_BASE / f"compare_se_amp_{fig_tag}.png"





def load_amp_series(amp_gd_dir: Path, return_std: bool = False):
    """
    Load AMP results aggregated over runs for each alpha.

    Returns
    -------
    alphas : (T,)
    u_mean : (T,)   mean of [0,0] entry across runs
    v_mean : (T,)   mean of [1,1] entry across runs
    uv_mean: (T,)   mean of [0,1] entry across runs
    vu_mean: (T,)   mean of [1,0] entry across runs
    (optional stds if return_std=True, same order as means)
    """
    alphas = np.load(amp_gd_dir / "alpha_values.npy")
    T = len(alphas)

    u_mean = np.full(T, np.nan); v_mean = np.full(T, np.nan)
    uv_mean = np.full(T, np.nan); vu_mean = np.full(T, np.nan)

    if return_std:
        u_std = np.full(T, np.nan); v_std = np.full(T, np.nan)
        uv_std = np.full(T, np.nan); vu_std = np.full(T, np.nan)

    for t, a in enumerate(alphas):
        tag = alabel(a)
        f = amp_gd_dir / f"final_overlaps_alpha_{tag}.npy"
        f_conv = amp_gd_dir / f"converged_alpha_{tag}.npy"
        if not f.exists():
            continue

        # O has shape (RUNS, 2, 2) — already "final over time"; we only average over runs
        O = np.load(f)  # (runs, 2, 2)
        mask = np.ones(O.shape[0], dtype=bool)

        if CONVERGED_ONLY and f_conv.exists():
            C = np.load(f_conv)  # (RUNS,)
            mask &= C.astype(bool)

        valid = np.isfinite(O.reshape(O.shape[0], -1)).all(axis=1)
        mask &= valid
        if not mask.any():
            continue

        
        M = np.nanmean(O[mask], axis=0)

        u_mean[t]  = M[0, 0]
        v_mean[t]  = M[1, 1]
        uv_mean[t] = M[0, 1]
        vu_mean[t] = M[1, 0]

        if return_std:
            u_std[t]  = np.nanstd(O[mask, 0, 0], ddof=1)
            v_std[t]  = np.nanstd(O[mask, 1, 1], ddof=1)
            uv_std[t] = np.nanstd(O[mask, 0, 1], ddof=1)
            vu_std[t] = np.nanstd(O[mask, 1, 0], ddof=1)

    if return_std:
        return alphas, u_mean, v_mean, uv_mean, vu_mean, u_std, v_std, uv_std, vu_std
    else:
        return alphas, u_mean, v_mean, uv_mean, vu_mean

def load_se_series(se_gd_dir: Path):
    """
    Load State Evolution results for each alpha.

    Each file q_list_alpha_<tag>.npy contains a list/array of 2x2 matrices (over iterations).
    We take the final matrix q_last and extract:
      - q[0,0]: overlap of u
      - q[1,1]: overlap of v
      - q[0,1]: off-diagonal (should equal q[1,0] for SE)

    Returns
    -------
    alpha_vals : (T,)
    ov_u       : (T,)
    ov_v       : (T,)
    ov_uv      : (T,)
    """
    files = sorted(se_gd_dir.glob("q_list_alpha_*.npy"))
    if not files:
        raise FileNotFoundError(f"No SE q_list_alpha_*.npy found under {se_gd_dir}")

    # recover alphas from filenames (reverse of alabel)
    def parse_alpha(fname: Path):
        tag = fname.stem.split("_alpha_")[-1]
        s = tag.replace("m", "-").replace("p", ".")
        try:
            return float(s)
        except ValueError:
            return None

    alpha_vals, ov_u, ov_v, ov_uv = [], [], [], []
    for f in files:
        a = parse_alpha(f)
        if a is None:
            continue
        q_list = np.load(f, allow_pickle=True)
        q_last = q_list[-1]
        alpha_vals.append(a)
        ov_u.append(q_last[0, 0])
        ov_v.append(q_last[1, 1])
        ov_uv.append(q_last[0, 1])  # off-diagonal

    # sort by alpha
    idx = np.argsort(alpha_vals)
    alpha_vals = np.array(alpha_vals)[idx]
    ov_u = np.array(ov_u)[idx]
    ov_v = np.array(ov_v)[idx]
    ov_uv = np.array(ov_uv)[idx]

    return alpha_vals, ov_u, ov_v, ov_uv


def main():
    # Load AMP (means over runs per alpha)
    a_amp, u_amp, v_amp, uv_amp, vu_amp = load_amp_series(AMP_GD_DIR, return_std=False)

    # Load SE (final q per alpha, including off-diagonal)
    a_se, u_se, v_se, uv_se = load_se_series(SE_GD_DIR)

    # Plot: u, v, and off-diagonal
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    suffix = " (converged only)" if CONVERGED_ONLY else ""
    fig.suptitle(TITLE + suffix)

    # --- u diag ---
    ax1.plot(a_se, u_se, linestyle="-", label=f"SE ({SE_MODE})")
    ax1.plot(a_amp, u_amp, linestyle="none", marker="x", label="AMP (mean)")
    ax1.set_ylabel("Overlap (u)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- v diag ---
    ax2.plot(a_se, v_se, linestyle="-", label=f"SE ({SE_MODE})")
    ax2.plot(a_amp, v_amp, linestyle="none", marker="x", label="AMP (mean)")
    ax2.set_ylabel("Overlap (v)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- off-diagonal ---
    # SE is symmetric: plot q[0,1] once as a solid line
    ax3.plot(a_se, uv_se, linestyle="-", label=f"SE off-diag q[0,1] ({SE_MODE})")

    # AMP may be asymmetric: show both entries as separate markers
    ax3.plot(a_amp, uv_amp, linestyle="none", marker="x", label="AMP mean O[0,1]")
    ax3.plot(a_amp, vu_amp, linestyle="none", marker="+", label="AMP mean O[1,0]")

    ax3.set_ylabel("Off-diagonal overlap")
    ax3.set_xlabel(r"$\alpha = n/d$")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if SAVE_PNG:
        fig.savefig(SAVE_PNG, dpi=150)
        print(f"Saved: {SAVE_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
