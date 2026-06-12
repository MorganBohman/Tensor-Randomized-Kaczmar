import numpy as np
import matplotlib.pyplot as plt
from ave_TRKB    import ave_tRKB, tprod
from F_TRKB      import FacTRKB
from ave_F_TRKB import ave_FacTRKB, tpinv


np.random.seed(42)

# tensor dimensions
m, r, n, l, p = 20, 8, 5, 5, 5

# factors U (m x r x p) and V (r x n x p), product A = UV (m x n x p)
U      = np.random.randn(m, r, p)
V      = np.random.randn(r, n, p)
X_true = np.random.randn(n, l, p)

# consistent system: Y = U * V * X_true
Y    = tprod(U, tprod(V, X_true))
X0   = np.zeros((n, l, p))
X_LN = tprod(tpinv(tprod(U, V)), Y)   # least-norm solution, needed by ave_FacTRKB

# block sizes and iterations
out_block = 5
in_block  = 4
num_its   = 3000


_, its_ave = ave_tRKB(tprod(U, V), Y, X0, num_its, out_block, alpha=1.0)
_, its_fac = FacTRKB(U, V, Y, X0, num_its, out_block, in_block)

# ave_FacTRKB returns 6 values: X, errs, res_errs, ln_errs, res_ln_errs, norms
_, _, res_ave_fac, _, _, _ = ave_FacTRKB(
    U, V, Y, X0, X_true, X_LN, num_its, out_block, in_block,
    alpha_u=1.0, alpha_v=1.0
)


A = tprod(U, V)

iters   = np.arange(num_its + 1)
res_ave = [np.linalg.norm(tprod(A, X) - Y) for X in its_ave]
res_fac = [np.linalg.norm(tprod(A, X) - Y) for X in its_fac]
# res_ave_fac already computed directly above — no iterates needed

plt.figure(figsize=(8, 5))
plt.semilogy(iters, res_ave,     'b-',  linewidth=2, label='ave_tRKB (inverse-free)')
plt.semilogy(iters, res_fac,     'r--', linewidth=2, label='FacTRKB (factorized)')
plt.semilogy(iters, res_ave_fac, 'g:',  linewidth=2, label='ave_FacTRKB (factorized + inverse-free)')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('variants_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
