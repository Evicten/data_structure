import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# -------------------- USER: set this to your run folder --------------------
# Example:
# TOP_DIR = Path("data/runs/betaU=1p000_betaV=2p000_qinit=0p100_0p100_0p010_samples=1000_iters=50_damp=0p500")
TOP_DIR = Path("runs") / "betaU=1p000_betaV=2p000_qinit=0p100_0p100_0p010_samples=1000_iters=50_damp=0p300"
# --------------------------------------------------------------------------

MODES = ["replica", "rBP"]
PAIRS = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]

# Figures will go under: figures/<RUN_TAG>/...
RUN_TAG = TOP_DIR.name
FIG_ROOT = Path("figures") / RUN_TAG

def outdir_for(mode: str, g: float, d: float) -> Path:
    """Directory to save figures for a specific (mode, gamma, delta)."""
    p = FIG_ROOT / f"mode={mode}" / f"gamma={alabel(g)}_delta={alabel(d)}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def alabel(a: float) -> str:
    return f"{a:.3f}".replace(".", "p")

def subdir_for(mode: str, g: float, d: float) -> Path:
    return TOP_DIR / f"mode={mode}" / f"gamma={alabel(g)}_delta={alabel(d)}"

def load_alpha_grid(subdir: Path) -> np.ndarray:
    f = subdir / "alpha_values.npy"
    if not f.exists():
        raise FileNotFoundError(f"alpha_values.npy not found in {subdir}")
    return np.load(f)

def qfile_for(subdir: Path, a: float) -> Path:
    tag = alabel(a)
    return subdir / f"q_list_alpha_{tag}.npy"

def load_final_q(subdir: Path, a: float) -> np.ndarray:
    f = qfile_for(subdir, a)
    arr = np.load(f)  # shape (iters, 2, 2)
    return arr[-1]    # final 2x2

def load_trajectory(subdir: Path, a: float) -> np.ndarray:
    f = qfile_for(subdir, a)
    return np.load(f)  # (iters, 2, 2)

def pick_alpha_index(alphas: np.ndarray, quantile: float = 0.8) -> int:
    # choose a “late-ish” alpha (80th percentile by default)
    idx = int(round(quantile * (len(alphas) - 1)))
    return max(0, min(idx, len(alphas) - 1))

def plot_q_vs_alpha(mode: str, g: float, d: float):
    subdir = subdir_for(mode, g, d)
    alphas = load_alpha_grid(subdir)

    q11, q22, q12 = [], [], []

    missing = 0
    for a in alphas:
        f = qfile_for(subdir, a)
        if not f.exists():
            missing += 1
            continue
        qf = load_final_q(subdir, a)
        q11.append(qf[0, 0])
        q22.append(qf[1, 1])
        q12.append(qf[0, 1])
        # enforce symmetry in case of tiny asymmetry
        # q12.append(0.5 * (qf[0, 1] + qf[1, 0]))

    if missing:
        print(f"[WARN] {mode} (γ={g:.3f}, δ={d:.3f}): {missing} alpha(s) missing files.")


    plt.figure(figsize=(8,5))
    plt.plot(alphas, q11, marker='o', lw=1.5, ms=3, label=r"$q_{11}$")
    plt.plot(alphas, q22, marker='o', lw=1.5, ms=3, label=r"$q_{22}$")
    plt.plot(alphas, q12, marker='o', lw=1.5, ms=3, label=r"$q_{12}$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$q$ values")
    plt.title(fr"{mode} | $\gamma={g:.1f}$, $\delta={d:.1f}$: final $q$ vs $\alpha$")
    plt.legend()
    out_dir = outdir_for(mode, g, d)
    out = out_dir / "q_vs_alpha.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[SAVE] {out}")


def plot_convergence(mode: str, g: float, d: float, quantile: float = 0.8):
    subdir = subdir_for(mode, g, d)
    alphas = load_alpha_grid(subdir)
    idx = pick_alpha_index(alphas, quantile=quantile)
    a = float(alphas[idx])

    traj = load_trajectory(subdir, a)  # (iters, 2, 2)
    iters = np.arange(traj.shape[0])

    q11 = traj[:, 0, 0]
    q22 = traj[:, 1, 1]
    # q12 = 0.5 * (traj[:, 0, 1] + traj[:, 1, 0])
    q12 = traj[:, 0, 1]

    plt.figure(figsize=(8,5))
    plt.plot(iters, q11, marker='.', lw=1.0, ms=3, label=r"$q_{11}$")
    plt.plot(iters, q22, marker='.', lw=1.0, ms=3, label=r"$q_{22}$")
    plt.plot(iters, q12, marker='.', lw=1.0, ms=3, label=r"$q_{12}$")
    plt.xlabel("Iteration")
    plt.ylabel(r"$q$ values")
    plt.title(fr"{mode} | $\gamma={g:.1f}$, $\delta={d:.1f}$: $q$ trajectory at $\alpha={a:.3f}$")
    plt.legend()
    out_dir = outdir_for(mode, g, d)
    out = out_dir / f"convergence_alpha_{alabel(a)}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[SAVE] {out}")


def main():
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        for (g, d) in PAIRS:
            plot_q_vs_alpha(mode, g, d)
            plot_convergence(mode, g, d, quantile=0.2)


if __name__ == "__main__":
    main()
