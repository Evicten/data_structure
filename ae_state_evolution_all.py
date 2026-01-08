import numpy as np
from scipy.stats import norm
from mpi4py import MPI
from lambda_nu_integrals import int_1, int_2, int_3, int_4, int_5, gamma_matrix


def beta_tilde(beta_v):
    return beta_v/(1+beta_v+np.sqrt(1+beta_v))

def state_func(state_hat, lam):
    m_hat, q_hat, V_hat = state_hat

    m = m_hat/(V_hat + lam)
    q = (q_hat+float(m_hat@m_hat.T))/(V_hat + lam)**2
    V = 1/(V_hat + lam)

    return (m, q, V)

def generate_latents(n, gamma=0, delta=1): 
    lambda_vals = np.random.randn(n) 
    unif = np.random.rand(n) 
    nu_vals = np.zeros(n, dtype=int) 

    mask1 = unif < gamma 
    nu_vals[mask1] = np.sign(lambda_vals[mask1]) 
    nu_vals[mask1][nu_vals[mask1]==0] = 1 #if lambda is exactly zero, set nu to 1 

    mask2 = (unif >= gamma) & (unif < gamma + delta) 
    t = norm.ppf(0.75) 
    nu_vals[mask2] = np.where(np.abs(lambda_vals[mask2]) > t, 1, -1) 

    mask3 = ~(mask1 | mask2) 
    nu_vals[mask3] = np.random.choice([-1, 1], size=np.sum(mask3), p=[1/2, 1/2]) 

    return lambda_vals, nu_vals

    
def F_1(beta_tilde_v):
    F1 = np.zeros((2,2))
    F1[1,1] = -beta_tilde_v

    return F1

def F_2(beta, lam_star, nu_star, Q, beta_tilde_v):
    beta_u, beta_v = beta
    F2 = np.zeros((2,))
    F2[0] = np.sqrt(beta_u)*lam_star
    F2[1] = np.sqrt(beta_v)*nu_star*(1-beta_tilde_v*Q[1,1])

    return F2

def F_2_batch(beta, lambda_vals, nu_vals, Q, beta_tilde_v):
    beta_u, beta_v = beta

    c0 = np.sqrt(beta_u)                             # scalar
    c1 = np.sqrt(beta_v) * (1 - beta_tilde_v * Q[1, 1])  # scalar

    # Each term is (n,), then we stack into (n, 2)
    F2_array = np.column_stack((
        c0 * lambda_vals,   # first coordinate for all samples
        c1 * nu_vals        # second coordinate for all samples
    ))

    return F2_array

def Z_out(beta_u, beta_v, gamma, delta, beta_tilde_v, omega, M, N, a, d):
    omega1, omega2 = omega
    b = np.sqrt(beta_u)*(omega1*N[0,0]+omega2*N[1,0])
    c = np.sqrt(beta_v)/(1-beta_tilde_v)*(omega1*N[0,1]+omega2*N[1,1])
    integral = int_1(a, b, c, d, gamma, delta)
    constant_piece = 1/(1-beta_tilde_v)*np.exp(-1/2*beta_v)*np.sqrt(np.linalg.det(N))*np.exp(-1/2*omega2**2*beta_v*(1-beta_v*M[1,1]))*np.exp(1/2*beta_v*(1+beta_v)*M[1,1])
    return constant_piece * integral

def Z_out_batch(beta_u, beta_v, gamma, delta, beta_tilde_v,
                omega, M, N, a, d):
    """
    omega: (S, 2)
    returns: (S, 1)
    """
    omega = np.asarray(omega)
    assert omega.ndim == 2 and omega.shape[1] == 2, "omega must be (samples, 2)"

    omega1 = omega[:, 0]
    omega2 = omega[:, 1]

    # Linear combinations
    b = np.sqrt(beta_u) * (omega @ N[:, 0])   # (S,)
    c = (np.sqrt(beta_v) / (1.0 - beta_tilde_v)) * (omega @ N[:, 1])  # (S,)

    # Integral term
    integral = int_1(a, b, c, d, gamma, delta)  # (S,)

    # Constant (scalar, reused)
    const = (
        1.0 / (1.0 - beta_tilde_v)
        * np.exp(-0.5 * beta_v)
        * np.sqrt(np.linalg.det(N))
        * np.exp(0.5 * beta_v * (1.0 + beta_v) * M[1, 1])
    )

    # Omega-dependent exponential
    omega_exp = np.exp(
        -0.5 * omega2**2 * beta_v * (1.0 - beta_v * M[1, 1])
    )  # (S,)

    Z = const * omega_exp * integral  # (S,)

    return Z[:, None]  # (S, 1)

def f_out(beta_u, beta_v, gamma, delta, beta_tilde_v, gamma_matrix, omega, N, a, d):
    omega1, omega2 = omega
    b = np.sqrt(beta_u)*(omega1*N[0,0]+omega2*N[1,0])
    c = np.sqrt(beta_v)/(1-beta_tilde_v)*(omega1*N[0,1]+omega2*N[1,1])
    denominator = int_1(a, b, c, d, gamma, delta)
    numerator_1 = np.sqrt(beta_u)*int_2(a, b, c, d, gamma, delta) 
    numerator_2 = np.sqrt(beta_v*(1+beta_v))*int_3(a, b, c, d, gamma, delta)
    numerator_vec = np.array([numerator_1, numerator_2])
    return N @ numerator_vec/denominator - N @ gamma_matrix @ omega

def f_out_batch(beta_u, beta_v, gamma, delta, beta_tilde_v, gamma_matrix,
                omega, N, a, d):
    """
    omega: (S, 2)
    N: (2, 2)
    gamma_matrix: (2, 2)
    returns: (S, 2)
    """
    omega = np.asarray(omega)
    assert omega.ndim == 2 and omega.shape[1] == 2, "omega must be (samples, 2)"

    # b = sqrt(beta_u) * (omega @ N[:,0])
    b = np.sqrt(beta_u) * (omega @ N[:, 0])

    # c = sqrt(beta_v)/(1-beta_tilde_v) * (omega @ N[:,1])
    c = (np.sqrt(beta_v) / (1.0 - beta_tilde_v)) * (omega @ N[:, 1])

    denom = int_1(a, b, c, d, gamma, delta)                 # (S,)
    num1  = np.sqrt(beta_u) * int_2(a, b, c, d, gamma, delta)  # (S,)
    num2  = np.sqrt(beta_v*(1.0+beta_v)) * int_3(a, b, c, d, gamma, delta)  # (S,)

    numerator_vec = np.stack([num1, num2], axis=1)          # (S, 2)

    term1 = (numerator_vec @ N.T) / denom[:, None]          # (S, 2)
    term2 = omega @ (gamma_matrix.T @ N.T)                  # (S, 2)   since N@gamma_matrix@omega_i

    return term1 - term2

def find_proximal_relu(omega, q, V):
    samples, _ = np.shape(omega)

    def prox_positive(omega, q, V):
        return omega/(1+V*(q-2)) #shape (samples, 1)
    
    def objective_min(h, omega, q, V):
        return -h*np.maximum(0, h) + 1/2*np.maximum(0,h)**2*q + 1/(2*V)*(h-omega)**2 #shape (samples, 1) 
    
    prox_neg = omega 
    prox_zero = np.zeros((samples, 1))
    prox_pos = prox_positive(omega, q, V)

    val_neg = objective_min(prox_neg, omega, q , V)
    val_zero = objective_min(prox_zero, omega, q , V)
    val_pos = objective_min(prox_pos, omega, q , V)

    vals = np.concatenate([val_neg, val_zero, val_pos], axis=1) 
    idx = np.argmin(vals, axis=1)                # (samples,)

    candidates = np.concatenate([prox_neg, prox_zero, prox_pos], axis=1)  # (samples, 3)
    rows = np.arange(samples)
    prox_opt = candidates[rows, idx]  # (samples,)

    return prox_opt[:, None]  # (samples, 1)

def find_proximal_linear_ae(omega, q, V):
    return omega/(1-V*(2-q)) #shape (samples, 1)



def hat_vars_MCMC_old(alpha, beta, q_vars, samples, gamma, delta, activation='relu'):
    beta_u, beta_v = beta
    m, q, V = q_vars
    beta_tilde_v = beta_tilde(beta_v)
    F1 = F_1(beta_tilde_v)

    sqrt_q = np.sqrt(q)
    Xi = np.random.randn(samples, 1)
    lam_vals, nu_vals = generate_latents(n=samples, gamma=gamma, delta=delta)

    F2 = F_2_batch(beta=beta, lambda_vals=lam_vals, nu_vals=nu_vals, Q=np.eye(2), beta_tilde_v=beta_tilde_v)

    h_star_mean = 1/np.sqrt(q)*Xi @ m # (samples, 2)
    h_star_covariance = np.eye(2) - 1/q * m.T @ m
    h_star_covariance_inv = np.linalg.inv(h_star_covariance)

    L = np.linalg.cholesky(h_star_covariance)
    Z = np.random.randn(samples, 2)
    h_star = h_star_mean + Z @ L.T #(samples, 2)

    omega = sqrt_q*Xi + (h_star @ F1 + F2) @ m.T 

    ### Here change for every activation

    if activation == 'relu':
        prox = find_proximal_relu(omega, q, V) 
        partial_q_loss = np.maximum(prox, 0)**2
    elif activation == 'linear':
        prox = find_proximal_linear_ae(omega, q, V)
        partial_q_loss = prox**2
    else:
        print("activation not found, check typos")

    ### END OF ACTIVATION SECTION (THIS SHOULD BE CHANGED TO A CLASS; MAKES MORES SENSE)

    h_star_centered = (h_star - 1/np.sqrt(q)*Xi@m)@h_star_covariance_inv # (samples, 2)
    data_model_mean = h_star @ F1 + F2 # (samples, 2)

    m_hat = alpha/V*np.mean((prox-omega)*(h_star_centered+data_model_mean), axis=0, keepdims=True)
    q_hat = alpha*np.mean((prox-omega)**2/V**2)
    V_hat = alpha*np.mean(partial_q_loss+1/V*(prox-omega)*(h_star_centered/q@m.T - (prox-omega)*Xi/np.sqrt(q)))

    return m_hat, q_hat, V_hat

def hat_vars_MCMC_new(alpha, beta, q_vars, samples, gamma, delta, activation='relu'):
    beta_u, beta_v = beta
    m, q, V = q_vars
    beta_tilde_v = beta_tilde(beta_v)
    Gamma = gamma_matrix(beta_v)
    V_star = np.eye(2) - 1/q*m.T @ m
    V_inv = np.linalg.inv(V_star)
    M = np.linalg.inv(V_inv + Gamma)
    N = V_inv @ M
    a = beta_u*(1-M[0,0])
    d = np.sqrt(beta_u*beta_v*(1+beta_v))*M[0,1]


    sqrt_q = np.sqrt(q)
    Xi = np.random.randn(samples, 1)
    omega = sqrt_q*Xi 
    omega_star = Xi@m/np.sqrt(q)

    # algo part
    if activation == 'relu':
        prox = find_proximal_relu(omega, q, V) 
        partial_q_loss = np.maximum(prox, 0)**2
    elif activation == 'linear':
        prox = find_proximal_linear_ae(omega, q, V)
        partial_q_loss = prox**2
    else:
        print("activation not found, check typos")
    
    fout = (prox - omega)/V # (samples, 1)

    # data part
    fout_star = f_out_batch(beta_u, beta_v, gamma, delta, beta_tilde_v, Gamma, omega_star, N, a, d) #(samples, 2)
    Zout_star = Z_out_batch(beta_u, beta_v, gamma, delta, beta_tilde_v, omega_star, M, N, a, d) #(samples, 1)
    

    m_hat = alpha*np.mean(Zout_star*fout*fout_star, axis=0, keepdims=True)
    q_hat = alpha*np.mean(Zout_star*fout**2)
    V_hat = alpha*np.mean(Zout_star*(np.maximum(prox, 0)**2 - 1/np.sqrt(q)*Xi*fout))+float(m_hat@m.T)/q

    return m_hat, q_hat, V_hat



def run_one_alpha(
    alpha,
    beta,
    gamma,
    delta,
    init,
    samples,
    iters,
    activation='relu',
    lam=1e-6,
    damping=0.7,
    print_every=50,
    version='old'
):
    """
    Run state evolution with MPI across ranks.
    """
    assert 0.0 <= gamma <= 1.0 and 0.0 <= delta <= 1.0 and gamma + delta <= 1.0, \
        "Require 0 ≤ γ, δ and γ+δ ≤ 1"


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    m0, q0, V0 = init
    state = (m0.copy(), float(q0), float(V0))

    if version == 'old':
        hat_vars_MCMC = hat_vars_MCMC_old
    elif version == 'new':
        hat_vars_MCMC = hat_vars_MCMC_new

    if rank == 0:
        state_list = []
        state_hat_list = []

    for i in range(iters):
        # Each rank computes its local q_hat
        state_hat_local = hat_vars_MCMC(alpha=alpha, beta=beta, q_vars=state, samples=samples, gamma=gamma, delta=delta, activation=activation)
        if rank != 0:
            # send local estimate to root, receive updated q
            comm.send(state_hat_local, dest=0)
            state = comm.recv(source=0)
            if any(x is None for x in state):
                break
            continue

        # --- root aggregates ---
        if (i % print_every) == 0:
            print(f"[alpha {alpha:.3f} | iter {i}] m, q, V =\n{state}")
            print(f"[alpha {alpha:.3f} | iter {i}] m_hat, q_hat, V_hat (local) =\n{state_hat_local}")

        # unpack local hats
        m_hat_local, q_hat_local, V_hat_local = state_hat_local

        # allocate arrays for all ranks, infer shapes from local
        m_hat_all = np.empty((size,) + m_hat_local.shape, dtype=np.float64)
        q_hat_all = np.empty((size,) + np.shape(q_hat_local), dtype=np.float64)
        V_hat_all = np.empty((size,) + np.shape(V_hat_local), dtype=np.float64)

        # store root's own result
        m_hat_all[0] = m_hat_local
        q_hat_all[0] = q_hat_local
        V_hat_all[0] = V_hat_local

        # receive from workers
        for j in range(1, size):
            m_hat_j, q_hat_j, V_hat_j = comm.recv(source=j)
            m_hat_all[j] = m_hat_j
            q_hat_all[j] = q_hat_j
            V_hat_all[j] = V_hat_j

        # average across ranks
        m_hat = np.mean(m_hat_all, axis=0)
        q_hat = np.mean(q_hat_all, axis=0)
        V_hat = np.mean(V_hat_all, axis=0)

        state_hat = (m_hat, q_hat, V_hat)

        (m, q, V) = state
        (m_new, q_new, V_new) = state_func(state_hat=state_hat, lam=lam)

        # Update q with damping
        state = (
            (1 - damping) * m_new + damping * m,
            (1 - damping) * q_new + damping * q,
            (1 - damping) * V_new + damping * V,
        )

        # store trajectories on root
        state_list.append(state)
        state_hat_list.append(state_hat)

        # broadcast updated q to workers
        for j in range(1, size):
            comm.send(state, dest=j)

    comm.Barrier()

    if rank == 0:
        return state_list, state_hat_list

    return None, None

def energetic_potential_MCMC(alpha, beta, q_vars, samples, gamma, delta, activation='relu'):

    beta_u, beta_v = beta
    m, q, V = q_vars
    beta_tilde_v = beta_tilde(beta_v)
    F1 = F_1(beta_tilde_v)

    sqrt_q = np.sqrt(q)
    Xi = np.random.randn(samples, 1)
    lam_vals, nu_vals = generate_latents(n=samples, gamma=gamma, delta=delta)

    F2 = F_2_batch(beta=beta, lambda_vals=lam_vals, nu_vals=nu_vals, Q=np.eye(2), beta_tilde_v=beta_tilde_v)

    h_star_mean = 1/np.sqrt(q)*Xi @ m # (samples, 2)
    h_star_covariance = np.eye(2) - 1/q * m.T @ m

    L = np.linalg.cholesky(h_star_covariance)
    Z = np.random.randn(samples, 2)
    h_star = h_star_mean + Z @ L.T #(samples, 2)

    omega = sqrt_q*Xi + (h_star @ F1 + F2) @ m.T 

    ### Here change for every activation

    if activation == 'relu':
        prox = find_proximal_relu(omega, q, V) 
        loss = - prox*np.maximum(0, prox) + 1/2*np.maximum(0, prox)**2*q
    elif activation == 'linear':
        prox = find_proximal_linear_ae(omega, q, V)
        loss = (q/2-1)*prox**2
    else:
        print("activation not found, check typos")

    ### END OF ACTIVATION SECTION (THIS SHOULD BE CHANGED TO A CLASS; MAKES MORES SENSE)
    psi_y = - alpha*(np.mean(loss) + 1/2*np.mean((prox-omega)**2)/V)

    return psi_y