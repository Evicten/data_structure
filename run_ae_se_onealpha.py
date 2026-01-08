import numpy as np
from mpi4py import MPI
from ae_state_evolution_correct import run_one_alpha

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ---- parameters ----
    alpha = 2.5
    beta = (1, 2)      # adjust if you want
    gamma = 0.0
    delta = 1.0
    samples = 20000         # per rank
    iters = 1000
    lam = 0.0             # as you asked
    damping = 0.9

    # initial state
    m0 = np.array([[0.001, 0.001]])
    q0 = 1.0
    V0 = 0.2
    q_init = (m0, q0, V0)

    # run SE
    state_traj, hat_traj = run_one_alpha(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        init=q_init,
        samples=samples,
        iters=iters,
        lam=lam,
        damping=damping,
        print_every=10,   # you can make this larger to reduce spam
    )

    if rank == 0:
        print("\n=== FINAL STATE (rank 0) ===")
        m_last, q_last, V_last = state_traj[-1]
        m_hat_last, q_hat_last, V_hat_last = hat_traj[-1]

        print("m_last:", m_last)
        print("q_last:", q_last)
        print("V_last:", V_last)
        print("m_hat_last:", m_hat_last)
        print("q_hat_last:", q_hat_last)
        print("V_hat_last:", V_hat_last)


if __name__ == "__main__":
    main()
