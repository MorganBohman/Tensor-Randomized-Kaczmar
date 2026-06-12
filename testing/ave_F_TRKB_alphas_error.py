import numpy as np
import matplotlib.pyplot as plt
from ave_FacTRKB import ave_FacTRKB, tprod, tpinv



np.random.seed(42)

m, r, n, l, p = 20, 8, 5, 5, 5

U      = np.random.randn(m, r, p)
V      = np.random.randn(r, n, p)
X_true = np.random.randn(n, l, p)
Y      = tprod(U, tprod(V, X_true))
X_LN   = tprod(tpinv(tprod(U, V)), Y)
X0     = np.zeros((n, l, p))

out_block = 5
in_block  = 4
num_its   = 3000
iters     = np.arange(num_its + 1)


X_final, errs, res_errs, ln_errs, res_ln_errs, norms = ave_FacTRKB(
    U, V, Y, X0, X_true, X_LN, num_its, out_block, in_block,
    alpha_u=1.0, alpha_v=1.0
)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

axes[0].semilogy(iters, res_errs,    'b-', linewidth=2)
axes[0].set_title('Residual error  ||UVX - Y||')
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Error')
axes[0].grid(True, which='both', alpha=0.3)

axes[1].semilogy(iters, errs,        'r-', linewidth=2)
axes[1].set_title('Relative solution error  ||X - X_true|| / ||X_true||')
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel('Error')
axes[1].grid(True, which='both', alpha=0.3)

axes[2].semilogy(iters, ln_errs,     'g-', linewidth=2)
axes[2].set_title('Relative least-norm error  ||X - X_LN|| / ||X_LN||')
axes[2].set_xlabel('Iterations')
axes[2].set_ylabel('Error')
axes[2].grid(True, which='both', alpha=0.3)

axes[3].semilogy(iters, res_ln_errs, 'm-', linewidth=2)
axes[3].set_title('Least-norm residual error  ||UV(X - X_LN)||')
axes[3].set_xlabel('Iterations')
axes[3].set_ylabel('Error')
axes[3].grid(True, which='both', alpha=0.3)

axes[4].plot(iters, norms,           'k-', linewidth=2)
axes[4].set_title('Iterate norm  ||X||')
axes[4].set_xlabel('Iterations')
axes[4].set_ylabel('Norm')
axes[4].grid(True, which='both', alpha=0.3)

axes[5].axis('off')

fig.suptitle('ave_FacTRKB — Full Error Tracking (alpha_u = alpha_v = 1.0)', fontsize=14)
fig.tight_layout()
plt.savefig('ave_FacTRKB_errors.png', dpi=150, bbox_inches='tight')



alphas = [0.1, 0.5, 1.0, 1.5, 2.0]
colors = ['purple', 'blue', 'green', 'orange', 'red']

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for alpha, color in zip(alphas, colors):
    _, errs_a, res_a, ln_a, res_ln_a, norms_a = ave_FacTRKB(
        U, V, Y, X0, X_true, X_LN, num_its, out_block, in_block,
        alpha_u=alpha, alpha_v=alpha
    )
    label = f'alpha = {alpha}'
    ax1.semilogy(iters, res_a, color=color, linewidth=2, label=label)
    ax2.semilogy(iters, ln_a,  color=color, linewidth=2, label=label)

ax1.set_xlabel('Iterations')
ax1.set_ylabel('Error')
ax1.set_title('Residual error  ||UVX - Y||')
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

ax2.set_xlabel('Iterations')
ax2.set_ylabel('Error')
ax2.set_title('Relative least-norm error  ||X - X_LN|| / ||X_LN||')
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

fig2.suptitle('ave_FacTRKB — Effect of alpha (alpha_u = alpha_v)', fontsize=14)
fig2.tight_layout()
plt.savefig('ave_FacTRKB_alpha_comparison.png', dpi=150, bbox_inches='tight')

plt.show()
