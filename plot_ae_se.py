import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======= USER SETTINGS (edit these by hand) =======

DATE = "20251211"     # <-- set to the date folder your run_ae_se.py used

# ---- SE parameters (must match the run that produced the data) ----
ACTIVATION = 'relu'
BETA_U = 1.0
BETA_V = 2.0
LAM    = 0.0          # SE_LAM

M0_U   = 0.001     # SE_M0_U
M0_V   = 0.001   # SE_M0_V
Q0     = 1.0         # SE_Q0
V0     = 0.1          # SE_V0

M0 = np.array([[M0_U, M0_V]])   # shape (1,2)
STATE_INIT = (M0, Q0, V0)

SAMPLES = 50000       # SE_SAMPLES (per rank)
ITERS   = 500       # SE_ITERS
DAMP    = 0.9     # SE_DAMPING

# --- Which (gamma, delta) to plot ---
GAMMA = 0.0
DELTA = 1.0

# ======= OUTPUT FIGURE PATH (auto-generated) =======

TITLE = "AE State Evolution — Cosine Similarity vs α (ReLU)"
FIG_BASE = Path("figures") / "ae_figures"
FIG_BASE.mkdir(parents=True, exist_ok=True)
XLIM = 10.0   # x-axis limit for alpha

# ===================================================

def alabel3(x: float, places: int = 3) -> str:
    """1.000 -> '1p000', 0.010 -> '0p010', 0.3 -> '0p300', -0.1 -> 'm0p100'."""
    s = f"{x:.{places}f}"
    return s.replace(".", "p").replace("-", "m")

# ----- PATH GENERATION: must mirror run_ae_se.py -----

BASE_OUT = Path("runs_ae")

def tag_init(state):
    m0, q0, V0 = state
    m0 = np.asarray(m0).ravel()
    return (
        f"init_m=({alabel3(m0[0])},{alabel3(m0[1])})_"
        f"q={alabel3(q0)}_V={alabel3(V0)}"
    )

def build_top_dir():
    parts = [
        f"act={ACTIVATION}",
        f"betaU={alabel3(BETA_U)}",
        f"betaV={alabel3(BETA_V)}",
        f"lambda={alabel3(LAM)}",
        tag_init(STATE_INIT),
        f"samples={SAMPLES}",
        f"iters={ITERS}",
        f"damp={alabel3(DAMP)}",
    ]
    return BASE_OUT / DATE / "_".join(parts)

TOP_DIR = build_top_dir()
GD_DIR  = TOP_DIR / f"gamma={alabel3(GAMMA)}_delta={alabel3(DELTA)}"

# Informative filename
fig_tag = "_".join([
    f"act={ACTIVATION}",
    f"gamma={alabel3(GAMMA)}",
    f"delta={alabel3(DELTA)}",
    f"betaU={alabel3(BETA_U)}",
    f"betaV={alabel3(BETA_V)}",
    f"lam={alabel3(LAM)}",
    tag_init(STATE_INIT),
    f"xlim={XLIM}",
])
SAVE_PNG = FIG_BASE / f"ae_se_cosine_{fig_tag}.png"

# ======= LOADING & COSINE COMPUTATION =======

def parse_alpha_from_filename(fname: Path) -> float | None:
    """
    Files are named like: state_list_alpha_0p700.npy
    We parse '0p700' -> 0.700
    """
    stem = fname.stem  # 'state_list_alpha_0p700'
    tag = stem.split("alpha_")[-1]  # '0p700'
    s = tag.replace("m", "-").replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None

def m_over_sqrt_q(m, q) -> np.ndarray:
    """
    Compute componentwise cosine-like overlap m / sqrt(q).

    - m: (1, 2) or (2,)
    - q: scalar or 0-D array

    Returns:
        (2,) array: [m_0/sqrt(q), m_1/sqrt(q)]
    """
    m_arr = np.asarray(m).ravel()          # shape (2,)
    q_scalar = float(q)                    # handles scalar or 0-D np.array
    return np.abs(m_arr) / np.sqrt(q_scalar)       # (2,)


def load_cosine_series(gd_dir: Path):
    """
    For each alpha, load the last state (m, q, V) from state_list_alpha_<tag>.npy,
    and compute cos(alpha) = m / sqrt(q) componentwise -> (2,) vector.
    """
    files = sorted(gd_dir.glob("state_list_alpha_*.npy"))
    if not files:
        raise FileNotFoundError(f"No state_list_alpha_*.npy found under {gd_dir}")

    alpha_vals = []
    cosine_vals = []   # will hold (2,) vectors

    for f in files:
        a = parse_alpha_from_filename(f)
        if a is None:
            continue

        state_traj = np.load(f, allow_pickle=True)
        # state_traj is an object array of (m, q, V) tuples
        m_last, q_last, V_last = state_traj[-1]

        cos_vec = m_over_sqrt_q(m_last, q_last)  # (2,)

        alpha_vals.append(a)
        cosine_vals.append(cos_vec)

    # sort by alpha
    idx = np.argsort(alpha_vals)
    alpha_vals = np.array(alpha_vals)[idx]
    cosine_vals = np.stack(cosine_vals, axis=0)[idx]  # shape (T, 2)

    return alpha_vals, cosine_vals


# ======= PLOTTING =======

def main():
    print(f"Using SE directory: {GD_DIR}")
    a_vals, cos_vals = load_cosine_series(GD_DIR)   # cos_vals shape (T, 2)

    np.save("data_send/first_alpha", a_vals)
    np.save("data_send/first_cos", cos_vals)

    plt.figure(figsize=(7, 5))
    plt.plot(a_vals, cos_vals[:, 0], 'o-', markersize=3, label=r'$m_1/\sqrt{q}$')
    plt.plot(a_vals, cos_vals[:, 1], 'o-', markersize=3, label=r'$m_2/\sqrt{q}$')

    plt.title(TITLE)
    plt.xlabel(r"$\alpha = n/d$")
    if XLIM is not None:
        plt.xlim(0, XLIM)
    plt.ylabel(r"$m_i/\sqrt{q}$")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {SAVE_PNG}")
    plt.show()



if __name__ == "__main__":
    main()
