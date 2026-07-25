"""
deblurring_FacTRAKB.py
------------------------
Deblurring using the true FacTRAKB (ave_FacTRKB) algorithm, matching
the paper's Section 5.2 experiment — doubly blurred images with a
factorized operator A = UV where:
  U = Gaussian blur operator  (5x5, sigma=2)
  V = Averaging blur operator (5x5)

The system is UVX = B, solved by alternating inverse-free Kaczmarz
steps on U (outer) and V (inner), as in ave_FacTRKB.

Requires: F_TRKB.py, ave_F_TRKB.py, mri_data.mat
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import convolve
from scipy.linalg import toeplitz
from F_TRKB     import tpinv, tran, tprod
from ave_F_TRKB import ave_FacTRKB


def gaussian_kernel(size=5, sigma=2.0):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()

def avg_kernel(size=5):
    return np.ones((size, size), dtype=float) / (size * size)

def circ_blurring_mxop(h, m, n):
    """Build (n x n x m) circulant blurring operator from filter h."""
    m1, n1 = h.shape
    h_padded = np.zeros((m, n))
    h_padded[:m1, :n1] = h
    A = np.zeros((n, n, m))
    for i in range(m):
        c = h_padded[i, :]
        r = np.concatenate([[c[0]], c[1:][::-1]])
        A[:, :, i] = toeplitz(c, r)
    return A

def reorder_tensor(X, m, n, p):
    """(m x n x p) image tensor → (n x p x m) for t-product."""
    X_unfold = np.zeros((m * n, p))
    for i in range(p):
        X_unfold[:, i] = X[:, :, i].reshape(-1, order='F')
    X_r = np.zeros((n, p, m))
    for i in range(m):
        X_r[:, :, i] = X_unfold[i*n:(i+1)*n, :]
    return X_r

def recover_img(Z, m, n, p):
    """(n x p x m) → (m x n x p) image tensor."""
    Z_unfold = np.zeros((m * n, p))
    for i in range(m):
        Z_unfold[i*n:(i+1)*n, :] = Z[:, :, i]
    Z_rec = np.zeros((m, n, p))
    for i in range(p):
        Z_rec[:, :, i] = Z_unfold[:, i].reshape(m, n, order='F')
    return Z_rec



print("Loading MRI data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data = loadmat(os.path.join(script_dir, 'mri_data.mat'))
X    = data['X']          # (128, 128, 12)
m, n, p_img = X.shape
print(f"  X: {X.shape}  min={X.min():.3f}  max={X.max():.3f}")



h_gauss = gaussian_kernel(5, 2.0)
h_avg   = avg_kernel(5)


Y = np.zeros_like(X)
for i in range(p_img):
    tmp    = convolve(X[:,:,i], h_gauss, mode='wrap')
    Y[:,:,i] = convolve(tmp,    h_avg,   mode='wrap')
print(f"  Y (doubly blurry): {Y.shape}")



print("Building factorized operator U, V...")
U = circ_blurring_mxop(h_gauss, m, n)
V = circ_blurring_mxop(h_avg,   m, n)
print(f"  U: {U.shape}  V: {V.shape}")

# verify A = UV gives the same blurring as sequential convolution
A = tprod(U, V)
print(f"  A = UV: {A.shape}")



X_r = reorder_tensor(X, m, n, p_img)
Y_r = reorder_tensor(Y, m, n, p_img)
print(f"  X_r: {X_r.shape}  Y_r: {Y_r.shape}")

# X_true and X_LN for metric tracking inside ave_FacTRKB
X_true = X_r.copy()
X_LN   = tprod(tpinv(A), Y_r)



T          = 10000
out_block  = 5
in_block   = 5
alpha_u    = 0.1
alpha_v    = 0.1

def run_factrakb(X0, label, seed):
    print(f"\nRunning FacTRAKB — {label} ({T} iters)...")
    np.random.seed(seed)
    X_final, errs, res_ln, inner_errs, res_inner_ln = ave_FacTRKB(
        U, V, Y_r, X0, X_true, X_LN, T, out_block, in_block,
        alpha_u=alpha_u, alpha_v=alpha_v
    )
    print(f"  final outer_rel={errs[-1]:.4e}  outer_res={res_ln[-1]:.4e}"
          f"  inner_rel={inner_errs[-1]:.4e}  inner_res={res_inner_ln[-1]:.4e}")
    return X_final, errs, res_ln, inner_errs, res_inner_ln

X0_zero   = np.zeros_like(X_r)
X0_blurry = Y_r.copy()
X0_rand   = np.random.default_rng(0).random(X_r.shape)

Z_zero_r,   e0, r0, ie0, ir0 = run_factrakb(X0_zero,   "zero init",   seed=42)
Z_blurry_r, eb, rb, ieb, irb = run_factrakb(X0_blurry, "blurry init", seed=123)
Z_rand_r,   er, rr, ier, irr = run_factrakb(X0_rand,   "rand init",   seed=7)

Z_zero   = recover_img(Z_zero_r,   m, n, p_img)
Z_blurry = recover_img(Z_blurry_r, m, n, p_img)
Z_rand   = recover_img(Z_rand_r,   m, n, p_img)
X_ln     = recover_img(X_LN,       m, n, p_img)



iters = np.arange(T + 1)

fig1, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].semilogy(iters, r0,  'b--', linewidth=2, label='zero init')
axes[0].semilogy(iters, rb,  'b-',  linewidth=2, label='blurry init')
axes[0].semilogy(iters, rr,  'b:',  linewidth=2, label='rand init')
axes[0].set_xlabel('Iteration', fontsize=13)
axes[0].set_ylabel('Outer residual  ||UV(X - X_LN)||', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, which='both', alpha=0.3)
axes[0].set_title('FacTRAKB — Outer System', fontsize=13)

axes[1].semilogy(iters, ir0, 'r--', linewidth=2, label='zero init')
axes[1].semilogy(iters, irb, 'r-',  linewidth=2, label='blurry init')
axes[1].semilogy(iters, irr, 'r:',  linewidth=2, label='rand init')
axes[1].set_xlabel('Iteration', fontsize=13)
axes[1].set_ylabel('Inner residual  ||V(X - X_LN)||', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, which='both', alpha=0.3)
axes[1].set_title('FacTRAKB — Inner System', fontsize=13)

fig1.tight_layout()
fig1.savefig('deblurring_FacTRAKB_residual.png', dpi=150, bbox_inches='tight')


nframes    = 4
row_labels = ['original', 'doubly blurry', 'zero init', 'blurry init', 'rand init', 'least norm']
images     = [X, Y, Z_zero, Z_blurry, Z_rand, X_ln]

fig2, axes2 = plt.subplots(6, nframes, figsize=(9, 14))
for row, (label, img) in enumerate(zip(row_labels, images)):
    for col in range(nframes):
        ax = axes2[row, col]
        ax.imshow(np.clip(img[:,:,col], 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col == 0:
            ax.set_ylabel(label, fontsize=10, rotation=90, labelpad=8)
        if row == 0:
            ax.set_title(f'frame{col+1}', fontsize=10)

fig2.tight_layout(pad=0.3)
fig2.savefig('deblurring_FacTRAKB_frames.png', dpi=150, bbox_inches='tight')

plt.show()
print("\nSaved deblurring_FacTRAKB_residual.png and deblurring_FacTRAKB_frames.png")
