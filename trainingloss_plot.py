import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======= USER SETTINGS (edit by hand) =======

# ---- SE / model parameters (MUST match the saved file) ----
ACTIVATION = "relu"

BETA_U = 1.0
BETA_V = 2.0
LAM    = 0.0

GAMMA  = 0.0
DELTA  = 1.0

M0_U   = 0.001
M0_V   = 0.001
Q0     = 1.0
V0     = 0.2

MC_SAMPLES = 100000

XLIM = None   # e.g. 10.0 or None
YLIM = None

TITLE = "Training loss vs α"

# ======= HELPERS =======

def alabel3(x: float, places: int = 3) -> str:
    s = f"{x:.{places}f}"
    return s.replace(".", "p").replace("-", "m")

def tag_init():
    return (
        f"init_m=({alabel3(M0_U)},{alabel3(M0_V)})_"
        f"q={alabel3(Q0)}_V={alabel3(V0)}"
    )

# ======= RECONSTRUCT TAG (must match saving script) =======

fig_tag = "_".join([
    f"act={ACTIVATION}",
    f"gamma={alabel3(GAMMA)}",
    f"delta={alabel3(DELTA)}",
    f"betaU={alabel3(BETA_U)}",
    f"betaV={alabel3(BETA_V)}",
    f"lam={alabel3(LAM)}",
    tag_init(),
    f"mc={MC_SAMPLES}",
])

DATA_BASE = Path("runs_ae") / "trainingloss_correct"
NPZ_PATH = DATA_BASE / f"ae_se_trainloss_{fig_tag}.npz"

# ======= LOAD & PLOT =======

def main():
    print(f"Loading data from: {NPZ_PATH}")
    data = np.load(NPZ_PATH)

    alpha = data["alpha"]
    loss  = data["train_loss"]

    # sort just in case
    idx = np.argsort(alpha)
    alpha = alpha[idx]
    loss  = loss[idx]

    plt.figure(figsize=(7, 5))
    plt.plot(alpha, loss, "o-", markersize=3)
    plt.xlabel(r"$\alpha = n/d$")
    plt.ylabel("training loss")
    plt.title(TITLE)
    plt.grid(True, alpha=0.4)

    if XLIM is not None:
        plt.xlim(0, XLIM)
    if YLIM is not None:
        plt.ylim(*YLIM)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
