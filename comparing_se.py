import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# USER SETTINGS (edit these)
# =========================

DATE = "20260107"

# shared params
ACTIVATION = "relu"   # only used in OLD naming (act=relu)
BETA_U = 1.0
BETA_V = 2.0
LAM    = 1.0

M0_U = 0.001
M0_V = 0.001
Q0   = 1.0

# only difference between the two runs:
V0_OLD = 0.2   # old run had V0=0.2
V0_NEW = 0.2   # new run had V0=0.1

SAMPLES = 50000
ITERS   = 1000
DAMP    = 0.9

GAMMA = 0.0
DELTA = 1.0

XLIM = 20.0  # set None to disable
TITLE = "AE State Evolution — OLD vs NEW"

BASE_OUT = Path("runs_ae")

# =========================
# Helpers
# =========================

def alabel3(x: float, places: int = 3) -> str:
    s = f"{x:.{places}f}"
    return s.replace(".", "p").replace("-", "m")

def tag_init(m0_u, m0_v, q0, v0):
    return (
        f"init_m=({alabel3(m0_u)},{alabel3(m0_v)})_"
        f"q={alabel3(q0)}_V={alabel3(v0)}"
    )

def build_top_dir(include_act: bool, v0: float) -> Path:
    parts = []
    if include_act:
        parts.append(f"act={ACTIVATION}")

    parts += [
        f"betaU={alabel3(BETA_U)}",
        f"betaV={alabel3(BETA_V)}",
        f"lambda={alabel3(LAM)}",
        tag_init(M0_U, M0_V, Q0, v0),
        f"samples={SAMPLES}",
        f"iters={ITERS}",
        f"damp={alabel3(DAMP)}",
    ]
    return BASE_OUT / DATE / "_".join(parts)

def gd_dir_from_top(top_dir: Path) -> Path:
    return top_dir / f"gamma={alabel3(GAMMA)}_delta={alabel3(DELTA)}"

def parse_alpha_from_filename(fname: Path) -> float | None:
    stem = fname.stem
    tag = stem.split("alpha_")[-1]
    s = tag.replace("m", "-").replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None

def m_over_sqrt_q(m, q) -> np.ndarray:
    m_arr = np.asarray(m).ravel()
    q_scalar = float(q)
    return np.abs(m_arr) / np.sqrt(q_scalar)

def load_series(gd_dir: Path):
    files = sorted(gd_dir.glob("state_list_alpha_*.npy"))
    if not files:
        raise FileNotFoundError(f"No state_list_alpha_*.npy found under {gd_dir}")

    alpha_vals, cos_vals, q_vals = [], [], []

    for f in files:
        a = parse_alpha_from_filename(f)
        if a is None:
            continue

        state_traj = np.load(f, allow_pickle=True)
        m_last, q_last, V_last = state_traj[-1]

        alpha_vals.append(a)
        cos_vals.append(m_over_sqrt_q(m_last, q_last))
        q_vals.append(float(q_last))

    idx = np.argsort(alpha_vals)
    alpha_vals = np.array(alpha_vals)[idx]
    cos_vals = np.stack(cos_vals, axis=0)[idx]
    q_vals = np.array(q_vals)[idx]
    return alpha_vals, cos_vals, q_vals

# =========================
# Main
# =========================

def main():
    TOP_OLD = build_top_dir(include_act=True,  v0=V0_OLD)
    TOP_NEW = build_top_dir(include_act=False, v0=V0_NEW)

    GD_OLD = gd_dir_from_top(TOP_OLD)
    GD_NEW = gd_dir_from_top(TOP_NEW)

    print("OLD GD_DIR:", GD_OLD)
    print("NEW GD_DIR:", GD_NEW)

    a_old, cos_old, q_old = load_series(GD_OLD)
    a_new, cos_new, q_new = load_series(GD_NEW)

    # cosine-like overlaps
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(a_old, cos_old[:, 0], "o-",  ms=3, label=r"OLD $|m_1|/\sqrt{q}$")
    plt.plot(a_old, cos_old[:, 1], "o-",  ms=3, label=r"OLD $|m_2|/\sqrt{q}$")
    plt.plot(a_new, cos_new[:, 0], "o--", ms=3, label=r"NEW $|m_1|/\sqrt{q}$")
    plt.plot(a_new, cos_new[:, 1], "o--", ms=3, label=r"NEW $|m_2|/\sqrt{q}$")
    plt.title(f"{TITLE} — cosine-like overlaps")
    plt.xlabel(r"$\alpha = n/d$")
    plt.ylabel(r"$|m_i|/\sqrt{q}$")
    if XLIM is not None:
        plt.xlim(0.0, XLIM)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # q vs alpha
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(a_old, q_old, "o-",  ms=3, label="OLD q")
    plt.plot(a_new, q_new, "o--", ms=3, label="NEW q")
    plt.title(f"{TITLE} — q vs α")
    plt.xlabel(r"$\alpha = n/d$")
    plt.ylabel(r"$q$")
    if XLIM is not None:
        plt.xlim(0.0, XLIM)
    plt.ylim(0.8, 1.0)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
