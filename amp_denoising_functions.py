import numpy as np
from lambda_nu_integrals import int_1, int_2, int_3, int_4, int_5

def beta_tilde(beta_v):
    return beta_v/(1+beta_v+np.sqrt(1+beta_v))

def c_2(beta_tilde_v, Q_22):
    return 1 - beta_tilde_v*Q_22

def c_1(c2, beta_tilde_v):
    return beta_tilde_v*(1+c2)/c2**2

def gamma_matrix(c1):
    mat = np.zeros((2,2))
    mat[1,1] = c1
    return mat

def p_out(beta_u, beta_v, gamma, delta, y, z, Q):
    if y == -1:
        return 1/2
    elif y == 1:
        beta_tilde_v = beta_tilde(beta_v)
        c2 = c_2(beta_tilde_v=beta_tilde_v, Q_22=Q[1,1])
        c1 = c_1(c2=c2, beta_tilde_v=beta_tilde_v)
        return 1/(2*np.abs(c2))*np.exp(-1/2*beta_v*Q[1,1])*np.exp(-1/2*c1*z[1]**2)*int_1(a=beta_u*(Q[0,0]+c1*Q[0,1]**2), b=np.sqrt(beta_u)*(z[0]+c1*Q[0,1]*z[1]), c=np.sqrt(beta_v)/c2*z[1], d=-np.sqrt(beta_u*beta_v)/c2*Q[0,1], gamma=gamma, delta=delta)


def Z_out(beta_v, gamma, delta, c1, c2, y, omega, Q, M, N, a, b, c, d):
    if y == -1:
        return 1/2
    elif y == 1:
        omega1, omega2 = omega
        integral = int_1(a, b, c, d, gamma, delta)
        constant_piece = 1/np.abs(c2)*np.exp(-1/2*beta_v*Q[1,1])*np.sqrt(np.linalg.det(N))*np.exp(-1/2*omega2**2*c1*(1-c1*M[1,1]))*np.exp(1/2*beta_v/(c2**2)*M[1,1])
        return constant_piece/2 * integral

def f_out(beta_u, beta_v, gamma, delta, gamma_matrix, c1, c2, y, omega, Q, N, a, b, c, d):
    if y == -1:
        return np.zeros((2,))
    elif y == 1:
        omega1, omega2 = omega
        denominator = int_1(a, b, c, d, gamma, delta)
        numerator_1 = np.sqrt(beta_u)*int_2(a, b, c, d, gamma, delta) 
        numerator_2 = np.sqrt(beta_v)/c2*int_3(a, b, c, d, gamma, delta)+np.sqrt(beta_u)*c1*Q[0,1]*int_2(a, b, c, d, gamma, delta)
        numerator_vec = np.array([numerator_1, numerator_2])
        return N @ numerator_vec/denominator - N @ gamma_matrix @ omega
    
def df_out(fout, beta_u, beta_v, gamma, delta, gamma_matrix, c1, c2, y, omega, Q, N, a, b, c, d):
    if y == -1:
        return np.zeros((2,2))
    if y == 1:
        int1 = int_1(a, b, c, d, gamma, delta)
        int4 = int_4(a, b, c, d, gamma, delta)
        int5 = int_5(a, b, c, d, gamma, delta)

        B = np.zeros((2,2))
        B[0,0] = beta_u*int5
        B[0,1] = np.sqrt(beta_u*beta_v)/c2*int4+beta_u*c1*Q[0,1]*int5
        B[1,0] = B[0,1]
        B[1,1] = beta_v/c2**2*int1 + beta_u*c1**2*Q[0,1]**2*int5+2*np.sqrt(beta_u*beta_v)*c1/c2*Q[0,1]*int4

        vec = fout + N @ gamma_matrix @ omega

        return 1/int1 * N @ B @ N.T - N @ gamma_matrix - np.outer(vec, vec)
    
def f_Q(beta_u, beta_v, gamma, delta, beta_tilde_v, c1, c2, y, omega, Q, M, V, a, b, c, d, fout, dfout):
    fQ = np.zeros((2,2))

    if y == -1:
        return fQ
    
    if y == 1:
        omega1, omega2 = omega

        int1 = int_1(a, b, c, d, gamma, delta)
        int2 = int_2(a, b, c, d, gamma, delta)
        int3 = int_3(a, b, c, d, gamma, delta)
        int4 = int_4(a, b, c, d, gamma, delta)
        int5 = int_5(a, b, c, d, gamma, delta)

        fQ[0,0] = - 1/2 * beta_u * int5 / int1
        fQ[0,1] = -1/(2*int1)*(np.sqrt(beta_u*beta_v)/c2*int4 + beta_u*c1*Q[0,1]*int5 - np.sqrt(beta_u)*c1*(np.sqrt(beta_u)*(M[0,1]+c1*Q[0,1]*M[1,1])*int5 + np.sqrt(beta_v)/c2*M[1,1]*int4 + (1- M[1,1]*c1)*omega2*int2))
        fQ[1,0] = fQ[0,1]
        #fQ11
        A = -1/2*beta_v + beta_tilde_v/c2 
        C = beta_tilde_v*np.sqrt(beta_v)/c2**2*(np.sqrt(beta_u)*(M[0,1]+c1*Q[0,1]*M[1,1])*int4 + np.sqrt(beta_v)/c2*M[1,1]*int1 + (1- M[1,1]*c1)*omega2*int3)
        D = np.sqrt(beta_u)*beta_tilde_v**2/c2**2*(1+2/c2)*Q[0,1]*(np.sqrt(beta_u)*(M[0,1]+c1*Q[0,1]*M[1,1])*int5+np.sqrt(beta_v)/c2*M[1,1]*int4+omega2*(1-M[1,1]*c1)*int2)
        E = - beta_tilde_v/c2**2 * np.sqrt(beta_u*beta_v)*Q[0,1]*int4
        F = - 1/2 * beta_u * beta_tilde_v**2/c2**2 * (1+2/c2)*Q[0,1]**2 * int5

        covZ = np.outer(omega + V @ fout, omega + V @ fout) + V @ dfout @ V + V
        B = - 1/2 * beta_tilde_v**2/c2**2*(1+2/c2)*covZ[1,1]

        fQ[1,1] = A + B + 1/int1*(C + D + E + F)

        return fQ

def integral_parameters(beta_u, beta_v, omega, Q, N, M, c1, c2):
    omega1, omega2 = omega

    a = beta_u*(Q[0,0]+c1*Q[0,1]**2-M[0,0]-2*c1*Q[0,1]*M[0,1]-c1**2*Q[0,1]**2*M[1,1])
    b = np.sqrt(beta_u)*(omega1*N[0,0]+omega2*N[1,0]+c1*Q[0,1]*(omega1*N[0,1]+omega2*N[1,1]))
    c = np.sqrt(beta_v)/c2*(omega1*N[0,1]+omega2*N[1,1])
    d = np.sqrt(beta_u*beta_v)/c2 * (-Q[0,1]+M[0,1]+c1*Q[0,1]*M[1,1])

    return a,b,c,d