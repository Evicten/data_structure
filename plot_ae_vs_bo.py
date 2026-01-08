import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======= USER SETTINGS (edit these to match your data folders) =======

# --- Common Parameters ---
DATE_AE = "20251211"
DATE_BO = "20251210"
BETA_U  = 1.0
BETA_V  = 2.0
GAMMA   = 0.0
DELTA   = 1.0
XLIM    = 10.0

# --- Autoencoder (AE) SE Settings ---
AE_SAMPLES = 50000
AE_ITERS   = 500
AE_DAMP    = 0.9
AE_LAM     = 0.0
AE_M0_U, AE_M0_V = 0.001, 0.001
AE_Q0, AE_V0     = 1.0, 0.1
AE_INIT    = (np.array([[AE_M0_U, AE_M0_V]]), AE_Q0, AE_V0)

# --- Bayes Optimal (BO) SE Settings ---
BO_MODE    = "replica"
BO_SAMPLES = 10000
BO_ITERS   = 50
BO_DAMP    = 0.3
BO_QINIT   = np.array([[0.001, 0.0001], [0.0001, 0.001]])

# ======= OUTPUT PATH =======
FIG_BASE = Path("figures") / "comparisons"
FIG_BASE.mkdir(parents=True, exist_ok=True)
SAVE_PNG = FIG_BASE / f"comparison_ae_vs_bo_g{GAMMA}_d{DELTA}.png"

# ===================================================

def alabel3(x: float, places: int = 3) -> str:
    s = f"{x:.{places}f}"
    return s.replace(".", "p").replace("-", "m")

# ----- PATH GENERATION -----

def get_ae_dir():
    m0 = np.asarray(AE_INIT[0]).ravel()
    init_tag = f"init_m=({alabel3(m0[0])},{alabel3(m0[1])})_q={alabel3(AE_INIT[1])}_V={alabel3(AE_INIT[2])}"
    parts = [
        f"betaU={alabel3(BETA_U)}", f"betaV={alabel3(BETA_V)}",
        f"lambda={alabel3(AE_LAM)}", init_tag,
        f"samples={AE_SAMPLES}", f"iters={AE_ITERS}", f"damp={alabel3(AE_DAMP)}"
    ]
    return Path("runs_ae") / DATE_AE / "_".join(parts) / f"gamma={alabel3(GAMMA)}_delta={alabel3(DELTA)}"

def get_bo_dir():
    q_tag = f"qinit={alabel3(BO_QINIT[0,0])}_{alabel3(BO_QINIT[1,1])}_{alabel3(BO_QINIT[0,1])}"
    parts = [
        f"betaU={alabel3(BETA_U)}", f"betaV={alabel3(BETA_V)}", q_tag,
        f"samples={BO_SAMPLES}", f"iters={BO_ITERS}", f"damp={alabel3(BO_DAMP)}"
    ]
    return Path("runs") / DATE_BO / "_".join(parts) / f"mode={BO_MODE}" / f"gamma={alabel3(GAMMA)}_delta={alabel3(DELTA)}"

# ======= LOADING FUNCTIONS =======

def parse_alpha(fname: Path) -> float:
    tag = fname.stem.split("alpha_")[-1]
    return float(tag.replace("m", "-").replace("p", "."))

def load_ae_data(path: Path):
    files = sorted(path.glob("state_list_alpha_*.npy"))
    if not files: raise FileNotFoundError(f"No AE data in {path}")
    
    alphas, u_overlap, v_overlap = [], [], []
    for f in files:
        a = parse_alpha(f)
        traj = np.load(f, allow_pickle=True)
        m_last, q_last, _ = traj[-1]
        m_arr = np.asarray(m_last).ravel()
        q_val = float(q_last)
        
        alphas.append(a)
        u_overlap.append(np.abs(m_arr[0]) / np.sqrt(q_val))
        v_overlap.append(np.abs(m_arr[1]) / np.sqrt(q_val))
    
    idx = np.argsort(alphas)
    return np.array(alphas)[idx], np.array(u_overlap)[idx], np.array(v_overlap)[idx]

def load_bo_data(path: Path):
    files = sorted(path.glob("q_list_alpha_*.npy"))
    if not files: raise FileNotFoundError(f"No BO data in {path}")
    
    alphas, u_overlap, v_overlap = [], [], []
    for f in files:
        a = parse_alpha(f)
        q_list = np.load(f, allow_pickle=True)
        q_last = q_list[-1]
        
        alphas.append(a)
        u_overlap.append(np.sqrt(q_last[0, 0]))
        v_overlap.append(np.sqrt(q_last[1, 1]))
        
    idx = np.argsort(alphas)
    return np.array(alphas)[idx], np.array(u_overlap)[idx], np.array(v_overlap)[idx]

# ======= MAIN PLOTTING =======

def main():
    # 1. Load Data
    try:
        a_ae, u_ae, v_ae = load_ae_data(get_ae_dir())
        a_bo, u_bo, v_bo = load_bo_data(get_bo_dir())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left Plot: U Overlap
    ax1.plot(a_bo, u_bo, 'k--', label=r'BO $\sqrt{q_{11}}$')
    ax1.plot(a_ae, u_ae, 'o-', markersize=4, label='AE $m_1/\sqrt{q}$', color='tab:blue')
    ax1.set_title(f"U-Overlap ($\\beta_u={BETA_U}$)")
    ax1.set_xlabel(r"$\alpha = n/d$")
    ax1.set_ylabel("Overlap")
    ax1.set_xlim(0, XLIM)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right Plot: V Overlap
    ax2.plot(a_bo, v_bo, 'k--', label=r'BO $\sqrt{q_{22}}$')
    ax2.plot(a_ae, v_ae, 's-', markersize=4, label='AE $m_2/\sqrt{q}$', color='tab:orange')
    ax2.set_title(f"V-Overlap ($\\beta_v={BETA_V}$)")
    ax2.set_xlabel(r"$\alpha = n/d$")
    ax2.set_ylabel("Overlap")
    ax2.set_xlim(0, XLIM)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"SE Comparison: AE vs Bayes Optimal ($\\gamma={GAMMA}, \\delta={DELTA}$)", fontsize=14)
    plt.tight_layout()

    # 3. Save and Show
    plt.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {SAVE_PNG}")
    plt.show()

if __name__ == "__main__":
    main()