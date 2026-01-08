import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ae_state_evolution_new import energetic_potential_MCMC

# ======= USER SETTINGS (edit these by hand) =======

DATE = "20260106"

BETA_U = 1.0
BETA_V = 2.0
LAM    = 0.0

M0_U   = 0.001
M0_V   = 0.001
Q0     = 1.0
V0     = 0.2

M0 = np.array([[M0_U, M0_V]])   # shape (1,2)
STATE_INIT = (M0, Q0, V0)

SAMPLES = 100000
ITERS   = 1000
DAMP    = 0.9

GAMMA = 0.0
DELTA = 1.0

# For Monte Carlo in the training loss
MC_SAMPLES = 100000

# ===================================================

def alabel3(x: float, places: int = 3) -> str:
    s = f"{x:.{places}f}"
    return s.replace(".", "p").replace("-", "m")

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

fig_tag = "_".join([
    f"gamma={alabel3(GAMMA)}",
    f"delta={alabel3(DELTA)}",
    f"betaU={alabel3(BETA_U)}",
    f"betaV={alabel3(BETA_V)}",
    f"lam={alabel3(LAM)}",
    tag_init(STATE_INIT),
    f"mc={MC_SAMPLES}",
])


# ===================== Loading helpers =====================

def parse_alpha_from_filename(fname: Path) -> float | None:
    stem = fname.stem
    tag = stem.split("alpha_")[-1]
    s = tag.replace("m", "-").replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None

def load_last_from_series(gd_dir: Path, prefix: str):
    """
    Generic loader for object-array trajectories saved as:
        <prefix>_alpha_<tag>.npy

    Each file must contain an object array where the last element is the last iterate.
    Returns dict: alpha -> last_item
    """
    files = sorted(gd_dir.glob(f"{prefix}_alpha_*.npy"))
    if not files:
        raise FileNotFoundError(f"No {prefix}_alpha_*.npy found under {gd_dir}")

    out = {}
    for f in files:
        a = parse_alpha_from_filename(f)
        if a is None:
            continue
        traj = np.load(f, allow_pickle=True)
        out[a] = traj[-1]
    return out

def load_order_params(gd_dir: Path):
    """
    Expected files:
      - state_list_alpha_*.npy  -> traj of (m, q, V)
      - hat_list_alpha_*.npy    -> traj of (m_hat, q_hat, V_hat)  (or whatever you named it)

    Adjust 'hat_list' prefix if your run script uses a different filename.
    """
    state_last = load_last_from_series(gd_dir, prefix="state_list")
    hat_last   = load_last_from_series(gd_dir, prefix="hat_list")  # <-- change if needed

    # intersect alphas present in both
    common_alphas = sorted(set(state_last.keys()).intersection(hat_last.keys()))
    if not common_alphas:
        raise RuntimeError("No common alphas between state_list and hat_list files.")

    m_list, q_list, V_list = [], [], []
    mhat_list, qhat_list, Vhat_list = [], [], []

    for a in common_alphas:
        m, q, V = state_last[a]
        m_hat, q_hat, V_hat = hat_last[a]

        m_list.append(np.asarray(m))
        q_list.append(float(q))
        V_list.append(float(V))

        mhat_list.append(np.asarray(m_hat))
        qhat_list.append(float(q_hat))
        Vhat_list.append(float(V_hat))

    return (
        np.array(common_alphas),
        np.array(m_list), np.array(q_list), np.array(V_list),
        np.array(mhat_list), np.array(qhat_list), np.array(Vhat_list),
    )


# ===================== Training loss skeleton =====================

def training_loss_from_order_params(alpha, 
    q_vars, q_hat, m_hat, V_hat,
    beta: tuple[float, float],
    lam: float,
    gamma: float,
    delta: float,
    samples: int = 200000
) -> float:
    """
    Skeleton: returns training loss = - free_entropy.

    Inputs:
      - m:      overlap vector (shape (1,2))
      - q, V:   scalars
      - m_hat:  conjugate overlap vector (shape (1,2))
      - q_hat, V_hat: scalars (conjugates)

    Replace the placeholders with your replica expression:
      free_entropy = "energetic part" + "entropic part" - "coupling terms"  (or similar)
    and return training_loss = -free_entropy.
    """

    beta_u, beta_v = beta
    m, q, V = q_vars
    # --- sanity checks (optional but helpful) ---
    if m.shape != m_hat.shape:
        raise ValueError(f"m shape {m.shape} != m_hat shape {m_hat.shape}")
    if q <= 0:
        raise ValueError(f"q must be > 0, got q={q}")


    # --- compute free entropy (replace with your replica expression) ---

    coupling = - (m @ m_hat.T)[0,0] - V * q_hat/2 + V_hat * q/2

    entropic = 1/(2*(lam + V_hat))*(q_hat+(m_hat@m_hat.T)[0,0])

    energetic = energetic_potential_MCMC(alpha, beta, q_vars, samples, gamma, delta)

    free_entropy = coupling + entropic + energetic

    training_loss = -free_entropy

    return training_loss

# ======= SAVE DATA (NO PLOTTING) =======

DATA_BASE = Path("runs_ae") / "trainingloss_new"
DATA_BASE.mkdir(parents=True, exist_ok=True)
SAVE_DATA = DATA_BASE / f"ae_se_trainloss_{fig_tag}.npz"

def main():
    print(f"Using SE directory: {GD_DIR}")

    (a_vals,
     m_list, q_vals, V_vals,
     mhat_list, qhat_vals, Vhat_vals) = load_order_params(GD_DIR)

    losses = np.empty_like(a_vals, dtype=float)
    beta = (BETA_U, BETA_V)

    for i, a in enumerate(a_vals):
        q_vars = m_list[i], q_vals[i], V_vals[i]
        losses[i] = training_loss_from_order_params(a, 
            q_vars,
            qhat_vals[i], mhat_list[i], Vhat_vals[i],
            beta=beta, lam=LAM,
            gamma=GAMMA, delta=DELTA,
            samples=MC_SAMPLES
        )

    # ---- save data only ----
    np.savez(
        SAVE_DATA,
        alpha=a_vals,
        train_loss=losses,
    )

    print(f"Saved data to: {SAVE_DATA}")


if __name__ == "__main__":
    main()
