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


def tpinv(A):
    n1, n2, n3 = A.shape
    A_hat = np.fft.fft(A, axis=2)
    A_pinv_hat = np.zeros((n2, n1, n3), dtype=complex)
    for i in range(n3):
        A_pinv_hat[:, :, i] = np.linalg.pinv(A_hat[:, :, i])
    return np.fft.ifft(A_pinv_hat, axis=2).real



def ave_FacTRKB(U, V, Y, X0, X_true, X_LN, T, out_block_size, in_block_size,
                              alpha_u=1.0, alpha_v=1.0, weighted=True):
  
    m_u = U.shape[0]
    m_v = V.shape[0]

    X = np.zeros_like(X0)
    Z = np.zeros((*tprod(V, X0).shape[:2], V.shape[2]))

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

    UV = tprod(U, V)

    def compute_metrics(X):
        err        = X - X_true
        ln_err     = X - X_LN
        res_err    = tprod(UV, X) - Y
        res_ln_err = tprod(UV, ln_err)
        return (
            np.linalg.norm(err.ravel())     / np.linalg.norm(X_true.ravel()),  # errs
            np.linalg.norm(res_err.ravel()),                                    # res_errs
            np.linalg.norm(ln_err.ravel())  / np.linalg.norm(X_LN.ravel()),    # ln_errs
            np.linalg.norm(res_ln_err.ravel()),                                 # res_ln_errs
            np.linalg.norm(X.ravel()),                                          # norms
        )

    e, r, l, rl, n = compute_metrics(X)
    errs        = [e]
    res_errs    = [r]
    ln_errs     = [l]
    res_ln_errs = [rl]
    norms       = [n]

    for _ in range(T):
        
        mu_t      = np.random.choice(m_u, size=out_block_size, replace=False, p=u_probs)
        U_block   = U[mu_t, :, :]
        Y_block   = Y[mu_t, :, :]
        U_block_t = tran(U_block)
        u_norm_sq = np.linalg.norm(U_block.ravel()) ** 2
        resid_y   = tprod(U_block, Z) - Y_block
        Z = Z - (alpha_u / u_norm_sq) * tprod(U_block_t, resid_y)

       
        nu_t      = np.random.choice(m_v, size=in_block_size, replace=False, p=v_probs)
        V_block   = V[nu_t, :, :]
        Z_block   = Z[nu_t, :, :]
        V_block_t = tran(V_block)
        v_norm_sq = np.linalg.norm(V_block.ravel()) ** 2
        resid_z   = tprod(V_block, X) - Z_block
        X = X - (alpha_v / v_norm_sq) * tprod(V_block_t, resid_z)

        
        e, r, l, rl, n = compute_metrics(X)
        errs.append(e)
        res_errs.append(r)
        ln_errs.append(l)
        res_ln_errs.append(rl)
        norms.append(n)

    return X, errs, res_errs, ln_errs, res_ln_errs, norms
