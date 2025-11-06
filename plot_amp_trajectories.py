# plot_amp_trajectories_for_alpha.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --- Common experiment parameters ---
DATE = "20251030"              # folder date stamp used in AMP path
BETA_U = 1.0
BETA_V = 2.0
D = 1000
TOL = 1e-4
DAMP = 0.6
MAX_ITER = 500
EPS = 1e-6
RUNS = 72
ALPHA_MIN = 0.1
ALPHA_MAX = 80
N_ALPHA = 50

# --- Which (gamma, delta, plant) to plot ---
GAMMA = 0.0
DELTA = 1.0
PLANT = 0.01



# ===================================================

def alabel(a: float) -> str:
    """Format numbers safely for folder names: 0.5 → '0p5', 1e-4 → '1p00000e-04'."""
    s = f"{a:.6g}"
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
ALPHA = 4.99184               # choose which alpha to visualize
SAVE_PNG = Path("amp_trajectories_alpha_5p0.png")  # set None to not save
# =============================

def alabel(a: float) -> str:
    return f"{a:.3f}".replace(".", "p").replace("-", "m")

def alabel4(a: float) -> str:
    return f"{a:.4f}".replace(".", "p").replace("-", "m")

def alabel5(a: float) -> str:
    return f"{a:.5f}".replace(".", "p").replace("-", "m")

def main():
    tag = alabel5(ALPHA)
    f_hist = AMP_GD_DIR / f"overlap_histories_alpha_{tag}.npy"
    if not f_hist.exists():
        raise FileNotFoundError(f"Not found: {f_hist}")

    # histories: object array length RUNS; each entry is (T_i, 2, 2)
    histories = np.load(f_hist, allow_pickle=True)

    # Determine max length to pad for median plotting
    T_max = 0
    for H in histories:
        if isinstance(H, np.ndarray):
            T_max = max(T_max, H.shape[0])

    # Prepare stacks for u and v (NaN-padded to T_max)
    U = np.full((len(histories), T_max), np.nan)
    V = np.full((len(histories), T_max), np.nan)

    for i, H in enumerate(histories):
        if not isinstance(H, np.ndarray) or H.shape == (0, 2, 2):
            continue
        T = H.shape[0]
        # diag overlaps over time
        U[i, :T] = H[:, 0, 0]
        V[i, :T] = H[:, 1, 1]

    # median (ignoring NaNs)
    u_med = np.nanmedian(U, axis=0)
    v_med = np.nanmedian(V, axis=0)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(fr"AMP Overlaps vs Iterations  (α={ALPHA:.3f})")

    # All runs as faint lines
    for i in range(U.shape[0]):
        ax1.plot(U[i], alpha=0.15)
        ax1.set_xlim(0, 50)
        ax2.plot(V[i], alpha=0.15)
        ax2.set_xlim(0, 50)

    # Median highlighted
    ax1.plot(u_med, linewidth=2)
    ax2.plot(v_med, linewidth=2)

    ax1.set_ylabel("Overlap (u)")
    ax2.set_ylabel("Overlap (v)")
    ax2.set_xlabel("Iteration")

    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if SAVE_PNG:
        fig.savefig(SAVE_PNG, dpi=150)
        print(f"Saved: {SAVE_PNG}")
    plt.show()

if __name__ == "__main__":
    main()
