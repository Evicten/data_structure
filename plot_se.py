
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======= USER SETTINGS (edit these by hand) =======
DATE = "20251210"

# --- SE (state evolution) parameters ---
SE_MODE = "replica"   # or "rBP"
SE_QINIT = np.array([[0.001, 0.0001],
                     [0.0001, 0.001]])
SE_SAMPLES = 10000
SE_ITERS = 50
SE_DAMP = 0.3

# --- Which (gamma, delta) to plot ---
GAMMA = 0.0
DELTA = 1.0

# --- Common experiment parameters (used only for tags/paths) ---
BETA_U = 1.0
BETA_V = 2.0

# ======= OUTPUT FIGURE PATH (auto-generated) =======
TITLE = "State Evolution — Overlaps vs α (u, v, off-diag) Correlated"
FIG_BASE = Path("figures")
FIG_BASE.mkdir(parents=True, exist_ok=True)
XLIM = 10

# ===================================================

def alabel3(x: float, places: int = 3) -> str:
    # 1.000 -> "1p000"; 0.010 -> "0p010"; 0.3 -> "0p300"
    s = f"{x:.{places}f}"
    return s.replace(".", "p").replace("-", "m")

# ======= SE PATH GENERATION =======

SE_BASE = Path("runs") / DATE
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

# Informative filename
fig_tag = "_".join([
    f"gamma={alabel3(GAMMA)}",
    f"delta={alabel3(DELTA)}",
    f"betaU={alabel3(BETA_U)}",
    f"betaV={alabel3(BETA_V)}",
    SE_MODE,
    SE_QTAG,
    f"xlim={XLIM}"
])
SAVE_PNG = FIG_BASE / f"se_overlaps_{fig_tag}.png"

# ======= LOADING =======

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

# ======= PLOTTING =======

def main():
    a_se, u_se, v_se, uv_se = load_se_series(SE_GD_DIR)

    plt.figure(figsize=(7, 5))
    plt.plot(a_se, u_se, 'o-', markersize=3, label=r'$q_{11}$')
    plt.plot(a_se, v_se, 'o-', markersize=3, label=r'$q_{22}$')
    plt.plot(a_se, uv_se, 'o-', markersize=3, label=r'$q_{12}$')
    # plt.axvline(x=26.25, linestyle='--', linewidth=2, color='black')  # 1st order PT

    plt.title(TITLE)
    plt.xlabel(r"$\alpha = n/d$")
    plt.xlim(0, XLIM)
    plt.ylabel(r"$q_{ij}$ values")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if SAVE_PNG:
        plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
        print(f"Saved: {SAVE_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
