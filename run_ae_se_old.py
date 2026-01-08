import numpy as np
from mpi4py import MPI
import os
from pathlib import Path
import json
import time

from ae_state_evolution_old import run_one_alpha

# ========= ENV / USER SETTINGS =========
# Defaults are here; env vars can override them.

def get_float_env(name, default):
    return float(os.getenv(name, str(default)))

def get_int_env(name, default):
    return int(os.getenv(name, str(default)))

def shp(x):
    if isinstance(x, np.ndarray): return x.shape
    if np.isscalar(x): return "scalar"
    try: return f"len={len(x)}"
    except: return type(x).__name_

# Model / algorithm params (can be overridden by env)
BETA_U = get_float_env("SE_BETA_U", 1.0)
BETA_V = get_float_env("SE_BETA_V", 2.0)
BETA   = (BETA_U, BETA_V)

SAMPLES = get_int_env("SE_SAMPLES", 2000) # samples per rank
ITERS   = get_int_env("SE_ITERS",   200)
DAMPING = get_float_env("SE_DAMPING", 0.4)

# Single (gamma, delta) and mode for this run
GAMMA = get_float_env("SE_GAMMA", 0.0)
DELTA = get_float_env("SE_DELTA", 1.0)

# Alpha grid (one grid per job)
ALPHA_MIN = get_float_env("SE_ALPHA_MIN", 0.1)
ALPHA_MAX = get_float_env("SE_ALPHA_MAX", 10.0)
N_ALPHA   = get_int_env("SE_N_ALPHA", 50)
ALPHA_VALUES = np.linspace(ALPHA_MIN, ALPHA_MAX, N_ALPHA)

# Regularisation
LAM = get_float_env("SE_LAM", 0.0)

# init entries (m, q, V) (upper triangle). Default: ([0.001, 0.001], 0.8, 0.7)
M0_U = get_float_env("SE_M0_U", 0.001)
M0_V = get_float_env("SE_M0_V", 0.001)
Q0   = get_float_env("SE_Q0", 0.8)
V0   = get_float_env("SE_V0", 0.2)

M0 = np.array([[M0_U, M0_V]])   # shape (1,2)
STATE_INIT = (M0, Q0, V0)

# Base output directory
BASE_OUT = Path("data/runs_ae")


# ==========================================

def alabel(a: float) -> str:
    """Safe tag for filenames: 0.500 -> '0p500'."""
    return f"{a:.3f}".replace(".", "p")

def tag_init(state):
    m0, q0, V0 = state
    m0 = np.asarray(m0).ravel()
    return (
        f"init_m=({alabel(m0[0])},{alabel(m0[1])})_"
        f"q={alabel(q0)}_V={alabel(V0)}"
    )

def build_top_dir() -> Path:
    """Top-level folder encodes the global params (same as your previous script)."""
    today = time.strftime("%Y%m%d")
    parts = [
        f"betaU={alabel(BETA_U)}",
        f"betaV={alabel(BETA_V)}",
        f"lambda={alabel(LAM)}",
        tag_init(STATE_INIT),
        f"samples={SAMPLES}",
        f"iters={ITERS}",
        f"damp={alabel(DAMPING)}",
    ]
    return BASE_OUT / today / "_".join(parts)

def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ---- create top-level directory and meta (root only)
    top_dir = build_top_dir()
    if rank == 0:
        top_dir.mkdir(parents=True, exist_ok=True)

    comm.Barrier()

    # ---- Subfolder for this (gamma, delta)
    sub_dir = top_dir / f"gamma={alabel(GAMMA)}_delta={alabel(DELTA)}"
    if rank == 0:
        sub_dir.mkdir(parents=True, exist_ok=True)
        # Save alpha grid once for this job
        np.save(sub_dir / "alpha_values.npy", ALPHA_VALUES)
        print(f"\n=== (gamma={GAMMA:.3f}, delta={DELTA:.3f}) ===")
    comm.Barrier()


    # Sweep over alpha
    for a in ALPHA_VALUES:
        tag = alabel(a)

        # Define per-alpha filenames
        f_state_list    = sub_dir / f"state_list_alpha_{tag}.npy"
        f_hat_list = sub_dir / f"hat_list_alpha_{tag}.npy"

        # Run one alpha (uses the chosen mode)
        state_traj, hat_traj = run_one_alpha(
            alpha=a,
            beta=BETA,
            gamma=GAMMA,
            delta=DELTA,
            init=STATE_INIT,
            samples=SAMPLES,
            iters=ITERS,
            lam=LAM,
            damping=DAMPING,
            print_every=50,   # you can make this larger to reduce spam
        )

        

        # Save (root only)
        if rank == 0:
            np.save(f_state_list, np.array(state_traj, dtype=object))
            np.save(f_hat_list,   np.array(hat_traj,   dtype=object))
            print(f"[ γ={GAMMA:.3f}, δ={DELTA:.3f}, α={a:.3f}] saved.")


    comm.Barrier()  # sync between (g, d) pairs inside this mode

if __name__ == "__main__":
    main()