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

def tRKB(A, B, X0, T, block_size, weighted=True):
    

    m = A.shape[0]
    X = X0.copy()
    its = [X.copy()]

    if block_size > m:
        raise ValueError(f"block_size ({block_size}) cannot exceed m ({m}).")

    if weighted:
        row_norms_sq = np.array([
            np.linalg.norm(A[i, :, :], 'fro') ** 2 for i in range(m)
        ])
        probs = row_norms_sq / row_norms_sq.sum()
    else:
        probs = None

    for _ in range(T):
        # sample without replacement so block rows are distinct
        i_t = np.random.choice(m, size=block_size, replace=False, p=probs)

        A_block = A[i_t, :, :]              # (block_size, n, p)
        B_block = B[i_t, :, :]              # (block_size, l, p)

        A_block_t      = tran(A_block)                       # (n, block_size, p)
        A_prod_inv     = tinv(tprod(A_block, A_block_t))     # (block_size, block_size, p)
        resid          = tprod(A_block, X) - B_block         # (block_size, l, p)

        X = X - tprod(A_block_t, tprod(A_prod_inv, resid))
        its.append(X.copy())

    return X, its
