import numpy as np
from pathlib import Path
from mpi4py import MPI
import json
import time

from state_evolution import run_one_alpha  # must accept mode="replica" or "rBP"

# ========= USER SETTINGS =========

# Exactly the three requested (gamma, delta) pairs:
PAIRS = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]

# Both modes to run for each pair:
MODES = ["replica", "rBP"]

# Alpha grid
ALPHA_VALUES = np.linspace(0.1, 20.0, 50)

# Model / algorithm params
BETA_U = 1
BETA_V = 2
BETA   = (BETA_U, BETA_V)

Q_INIT  = np.array([[0.10, 0.01],
                    [0.01, 0.10]], dtype=float)

SAMPLES = 1000
ITERS   = 50
DAMPING = 0.3

# Base output directory
BASE_OUT = Path("data/runs")

# ==========================================

def alabel(a: float) -> str:
    """Safe tag for filenames: 0.500 -> '0p500'."""
    return f"{a:.3f}".replace(".", "p")

def tag_qinit(q):
    """Encode q_init upper triangle (q11, q22, q12) into a short tag."""
    return f"qinit={alabel(q[0,0])}_{alabel(q[1,1])}_{alabel(q[0,1])}"

def build_top_dir() -> Path:
    """Top-level folder encodes the global params (same as your previous script)."""
    today = time.strftime("%Y%m%d")
    parts = [
        f"betaU={alabel(BETA_U)}",
        f"betaV={alabel(BETA_V)}",
        tag_qinit(Q_INIT),
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
        top_meta = dict(
            beta_u=BETA_U, beta_v=BETA_V,
            q_init=Q_INIT.tolist(),
            samples=SAMPLES, iters=ITERS,
            damping=DAMPING,
            modes=MODES,
            note="Global params constant across all (gamma, delta) and modes."
        )
        write_json(top_dir / "meta.json", top_meta)
    comm.Barrier()

    total_jobs = len(PAIRS) * len(MODES)

    job_idx = 0
    for mode in MODES:
        # Mode-level folder
        mode_dir = top_dir / f"mode={mode}"
        if rank == 0:
            mode_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        for (g, d) in PAIRS:
            job_idx += 1

            # Subfolder per (gamma, delta) within mode
            sub_dir = mode_dir / f"gamma={alabel(g)}_delta={alabel(d)}"
            if rank == 0:
                sub_dir.mkdir(parents=True, exist_ok=True)
                # Save the alpha grid once per (gamma,delta,mode)
                np.save(sub_dir / "alpha_values.npy", ALPHA_VALUES)
                print(f"\n=== [{job_idx}/{total_jobs}] mode={mode} "
                      f"(gamma={g:.3f}, delta={d:.3f}) ===")
            comm.Barrier()

            # Sweep over alpha
            for a in ALPHA_VALUES:
                tag = alabel(a)

                # Define per-alpha filenames
                f_q_list    = sub_dir / f"q_list_alpha_{tag}.npy"
                f_qhat_list = sub_dir / f"q_hat_list_alpha_{tag}.npy"

                # Run one alpha (uses the chosen mode)
                q_list, q_hat_list = run_one_alpha(
                    alpha=a, beta=BETA, gamma=g, delta=d,
                    q_init=Q_INIT, samples=SAMPLES, iters=ITERS,
                    damping=DAMPING, mode=mode
                )

                # Save (root only)
                if rank == 0:
                    np.save(f_q_list,    q_list)
                    np.save(f_qhat_list, q_hat_list)
                    print(f"[mode={mode} | γ={g:.3f}, δ={d:.3f}, α={a:.3f}] saved.")

            comm.Barrier()  # sync between (g, d) pairs inside this mode

    comm.Barrier()

if __name__ == "__main__":
    main()
