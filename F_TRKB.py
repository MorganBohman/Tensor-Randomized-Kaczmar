import numpy as np


def tprod(A, B):
    n1, n2, n3 = A.shape
    m1, m2, m3 = B.shape
    if n3 != m3:
        raise ValueError(f"Third dimensions must match: A has {n3}, B has {m3}.")
    if n2 != m1:
        raise ValueError(f"Inner dimensions must match: A has {n2} cols, B has {m1} rows.")
    A_hat = np.fft.fft(A, axis=2)
    B_hat = np.fft.fft(B, axis=2)
    C_hat = np.zeros((n1, m2, n3), dtype=complex)
    for i in range(n3):
        C_hat[:, :, i] = A_hat[:, :, i] @ B_hat[:, :, i]
    return np.fft.ifft(C_hat, axis=2).real


def tran(A):
    n1, n2, n3 = A.shape
    AT = np.zeros((n2, n1, n3), dtype=A.dtype)
    AT[:, :, 0] = A[:, :, 0].T
    for i in range(1, n3):
        AT[:, :, i] = A[:, :, n3 - i].T
    return AT


def tinv(A):
    n1, n2, n3 = A.shape
    if n1 != n2:
        raise ValueError(f"Frontal slices must be square; got ({n1}, {n2}).")
    A_hat = np.fft.fft(A, axis=2)
    A_hat_inv = np.zeros_like(A_hat)
    for i in range(n3):
        A_hat_inv[:, :, i] = np.linalg.inv(A_hat[:, :, i])
    return np.fft.ifft(A_hat_inv, axis=2).real



def FacTRKB(U, V, Y, X0, T, out_block_size, in_block_size, weighted=True):
    m_u = U.shape[0]    
    m_v = V.shape[0]    
    X   = X0.copy()
    Z   = np.zeros((*tprod(V, X0).shape[:2], V.shape[2])) 
    its = [X.copy()]

    if out_block_size > m_u:
        raise ValueError(f"out_block_size ({out_block_size}) cannot exceed rows of U ({m_u}).")
    if in_block_size > m_v:
        raise ValueError(f"in_block_size ({in_block_size}) cannot exceed rows of V ({m_v}).")

   
    if weighted:
        u_norms_sq = np.array([np.linalg.norm(U[i,:,:], 'fro')**2 for i in range(m_u)])
        v_norms_sq = np.array([np.linalg.norm(V[i,:,:], 'fro')**2 for i in range(m_v)])
        u_probs = u_norms_sq / u_norms_sq.sum()
        v_probs = v_norms_sq / v_norms_sq.sum()
    else:
        u_probs = v_probs = None

    for _ in range(T):
       #outer
        mu_t    = np.random.choice(m_u, size=out_block_size, replace=False, p=u_probs)
        U_block = U[mu_t, :, :]                          # (out_block_size, r, p)
        Y_block = Y[mu_t, :, :]                          # (out_block_size, l, p)
        U_block_t   = tran(U_block)                      # (r, out_block_size, p)
        U_prod_inv  = tinv(tprod(U_block, U_block_t))    # (out_block_size, out_block_size, p)
        resid_y     = tprod(U_block, Z) - Y_block        # (out_block_size, l, p)
        Z = Z - tprod(U_block_t, tprod(U_prod_inv, resid_y))

        #inner
        nu_t    = np.random.choice(m_v, size=in_block_size, replace=False, p=v_probs)
        V_block = V[nu_t, :, :]                          # (in_block_size, n, p)
        Z_block = Z[nu_t, :, :]                          # (in_block_size, l, p)
        V_block_t   = tran(V_block)                      # (n, in_block_size, p)
        V_prod_inv  = tinv(tprod(V_block, V_block_t))    # (in_block_size, in_block_size, p)
        resid_z     = tprod(V_block, X) - Z_block        # (in_block_size, l, p)
        X = X - tprod(V_block_t, tprod(V_prod_inv, resid_z))

        its.append(X.copy())

    return X, its
