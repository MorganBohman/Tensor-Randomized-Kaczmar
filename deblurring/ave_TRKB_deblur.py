import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import convolve
from scipy.linalg import toeplitz
from ave_F_TRKB import tprod
from ave_F_TRKB import tpinv, tran

def circ_blurring_mxop(h, m, n):
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
    X_unfold = np.zeros((m * n, p))
    for i in range(p):
        X_unfold[:, i] = X[:, :, i].reshape(-1, order='F')
    X_r = np.zeros((n, p, m))
    for i in range(m):
        X_r[:, :, i] = X_unfold[i*n:(i+1)*n, :]
    return X_r

def recover_img(Z, m, n, p):
    Z_unfold = np.zeros((m * n, p))
    for i in range(m):
        Z_unfold[i*n:(i+1)*n, :] = Z[:, :, i]
    Z_rec = np.zeros((m, n, p))
    for i in range(p):
        Z_rec[:, :, i] = Z_unfold[:, i].reshape(m, n, order='F')
    return Z_rec

def run_deblur(A, Y_r, X0, X_r, T, block_size, alpha, label):
    m_a = A.shape[0]
    X = X0.copy()
    row_norms_sq = np.array([np.linalg.norm(A[i,:,:], 'fro')**2 for i in range(m_a)])
    probs = row_norms_sq / row_norms_sq.sum()

    res_errs  = [np.linalg.norm(tprod(A, X) - Y_r)]
    log_iters = [0]

    for t in range(T):
        i_t       = np.random.choice(m_a, size=block_size, replace=False, p=probs)
        A_block   = A[i_t, :, :]
        B_block   = Y_r[i_t, :, :]
        A_block_t = tran(A_block)
        nsq       = np.linalg.norm(A_block.ravel()) ** 2
        resid     = tprod(A_block, X) - B_block
        X         = X - (alpha / nsq) * tprod(A_block_t, resid)

        if (t + 1) % 500 == 0:
            r = np.linalg.norm(tprod(A, X) - Y_r)
            res_errs.append(r)
            log_iters.append(t + 1)
            print(f"  [{label}] iter {t+1}/{T}  res={r:.4e}")

    res_errs.append(np.linalg.norm(tprod(A, X) - Y_r))
    log_iters.append(T)
    return X, res_errs, log_iters



data = loadmat(r'C:\Users\Admin\OneDrive\Desktop\KACZMAR\mri_data.mat')
X    = data['X']                          # (128, 128, 12), values in [0,1]
m, n, p = X.shape
print(f"  X: {X.shape}  min={X.min():.3f}  max={X.max():.3f}")



def gaussian_kernel(size=5, sigma=2.0):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()

h = gaussian_kernel(5, 2.0)
Y = np.stack([convolve(X[:,:,i], h, mode='wrap') for i in range(p)], axis=2)
print(f"  Y (blurry): {Y.shape}")


A   = circ_blurring_mxop(h, m, n)
X_r = reorder_tensor(X, m, n, p)
Y_r = reorder_tensor(Y, m, n, p)
print(f"  A: {A.shape}  X_r: {X_r.shape}")



T          = 10000
block_size = 5
alpha      = 0.1     # stable value for 128x128 Gaussian blur

np.random.seed(42)
print(f"\nRunning zero init ({T} iters)...")
Z_zero_r, res_zero, iters_zero = run_deblur(
    A, Y_r, np.zeros_like(X_r), X_r, T, block_size, alpha, "zero init")

np.random.seed(42)
print(f"Running blurry init ({T} iters)...")
Z_blurry_r, res_blurry, iters_blurry = run_deblur(
    A, Y_r, Y_r.copy(), X_r, T, block_size, alpha, "blurry init")

Z_zero   = recover_img(Z_zero_r,   m, n, p)
Z_blurry = recover_img(Z_blurry_r, m, n, p)
print("\nComputing least-norm solution...")
X_ln = recover_img(tprod(tpinv(A), Y_r), m, n, p)



fig1, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(iters_zero,   res_zero,   'b--', linewidth=2, label='zero init Ave_TRKB')
ax.semilogy(iters_blurry, res_blurry, 'b-',  linewidth=2, label='blurry init Ave_TRKB')
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Residual Error  ||AX - Y||', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, which='both', alpha=0.3)
fig1.tight_layout()
fig1.savefig('deblurring_residual.png', dpi=150, bbox_inches='tight')



nframes    = 4
row_labels = ['original', 'blurry', 'zero init', 'blurry init', 'least norm']
images     = [X, Y, Z_zero, Z_blurry, X_ln]

fig2, axes = plt.subplots(5, nframes, figsize=(9, 12))
for row, (label, img) in enumerate(zip(row_labels, images)):
    for col in range(nframes):
        ax = axes[row, col]
        ax.imshow(np.clip(img[:,:,col], 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col == 0:
            ax.set_ylabel(label, fontsize=11, rotation=90, labelpad=8)
        if row == 0:
            ax.set_title(f'frame{col+1}', fontsize=11)

fig2.tight_layout(pad=0.3)
fig2.savefig('deblurring_frames.png', dpi=150, bbox_inches='tight')

plt.show()
print("Saved deblurring_residual.png and deblurring_frames.png")
