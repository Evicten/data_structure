import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ======= USER SETTINGS (edit these by hand) =======

# Which run to read (must match how you saved)
DATE       = "20251114"   # or "latest" if you want to auto-pick newest date folder
SE_MODE    = "replica"    # or "rBP"
GAMMA      = 0.0
DELTA      = 1.0

# Must match the run's top-dir tags
BETA_U     = 1.0
BETA_V     = 2.0
SE_QINIT   = np.array([[0.001, 0.0001],
                       [0.0001, 0.001]], dtype=float)
SE_SAMPLES = 5000
SE_ITERS   = 100
SE_DAMP    = 0.3

# Which alpha to target (we'll pick the closest saved one)
ALPHA_TARGET = 1.2

# Paths
BASE_OUT  = Path("runs")
FIG_BASE  = Path("figures")
FIG_BASE.mkdir(parents=True, exist_ok=True)

# ===================================================

def alabel(x: float, places: int = 3) -> str:
    """Tag-friendly number: 0.500 -> '0p500' (keep '-' as is)."""
    s = f"{x:.{places}f}"
    return s.replace(".", "p")

def tag_qinit(q: np.ndarray) -> str:
    return f"qinit={alabel(q[0,0])}_{alabel(q[1,1])}_{alabel(q[0,1])}"

def pick_date_folder(base: Path, date: str | None) -> Path:
    if date and date != "latest":
        d = base / date
        if not d.is_dir():
            raise FileNotFoundError(f"DATE folder not found: {d}")
        return d
    # auto-pick latest YYYYMMDD
    candidates = [p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)]
    if not candidates:
        raise FileNotFoundError(f"No date folders found under {base} (expected YYYYMMDD).")
    return sorted(candidates, key=lambda p: p.name)[-1]

def build_top_dir() -> Path:
    parts = [
        f"betaU={alabel(BETA_U)}",
        f"betaV={alabel(BETA_V)}",
        tag_qinit(SE_QINIT),
        f"samples={SE_SAMPLES}",
        f"iters={SE_ITERS}",
        f"damp={alabel(SE_DAMP)}",
    ]
    date_dir = pick_date_folder(BASE_OUT, DATE)
    return date_dir / "_".join(parts)

def se_gd_dir() -> Path:
    top = build_top_dir()
    return top / f"mode={SE_MODE}" / f"gamma={alabel(GAMMA)}_delta={alabel(DELTA)}"

def parse_alpha_from_fname(fname: Path) -> float | None:
    # q_list_alpha_<tag>.npy, where <tag> is '0p500' or '-0p500'
    tag = fname.stem.split("_alpha_")[-1]
    try:
        return float(tag.replace("p", "."))
    except ValueError:
        return None

def find_closest_alpha_file(se_dir: Path, alpha_target: float) -> tuple[Path, float]:
    files = sorted(se_dir.glob("q_list_alpha_*.npy"))
    if not files:
        raise FileNotFoundError(f"No q_list_alpha_*.npy under {se_dir}")
    alphas, good_files = [], []
    for f in files:
        a = parse_alpha_from_fname(f)
        if a is None:
            continue
        alphas.append(a)
        good_files.append(f)
    if not alphas:
        raise RuntimeError(f"No parsable alpha tags in {se_dir}")
    alphas = np.array(alphas)
    idx = int(np.argmin(np.abs(alphas - alpha_target)))
    return good_files[idx], float(alphas[idx])

def main():
    se_dir = se_gd_dir()
    f_q_list, alpha_chosen = find_closest_alpha_file(se_dir, ALPHA_TARGET)

    q_list = np.load(f_q_list, allow_pickle=True)  # shape: (iters, 2, 2)
    q_list = np.asarray(q_list)
    iters = np.arange(len(q_list))

    q11 = q_list[:, 0, 0]
    q22 = q_list[:, 1, 1]
    q12 = q_list[:, 0, 1]

    title = (r"State Evolution — $q_{ij}$ vs iteration"
             f"\nmode={SE_MODE}, γ={GAMMA:.3f}, δ={DELTA:.3f}, α*={alpha_chosen:.3f}"
             f"  (target={ALPHA_TARGET:.3f})")
    date_used = pick_date_folder(BASE_OUT, DATE).name
    fig_tag = "_".join([
        f"gamma={alabel(GAMMA)}",
        f"delta={alabel(DELTA)}",
        f"betaU={alabel(BETA_U)}",
        f"betaV={alabel(BETA_V)}",
        SE_MODE,
        tag_qinit(SE_QINIT),
        f"alphaChosen={alabel(alpha_chosen)}",
        f"date={date_used}",
    ])
    save_png = FIG_BASE / f"se_q_vs_iter_{fig_tag}.png"

    plt.figure(figsize=(7, 5))
    plt.plot(iters, q11, 'o-', markersize=3, label=r'$q_{11}$')
    plt.plot(iters, q22, 'o-', markersize=3, label=r'$q_{22}$')
    plt.plot(iters, q12, 'o-', markersize=3, label=r'$q_{12}$')
    plt.xlabel("Iteration")
    plt.ylabel(r"$q_{ij}$")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_png, dpi=150, bbox_inches="tight")
    print(f"Loaded file: {f_q_list}")
    print(f"Target α={ALPHA_TARGET:.3f} -> chosen α={alpha_chosen:.3f}")
    print(f"Saved: {save_png}")
    plt.show()

if __name__ == "__main__":
    main()
