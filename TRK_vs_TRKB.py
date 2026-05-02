
import numpy as np
import matplotlib.pyplot as plt
from TRK  import tRK, tprod
from TRKB import tRKB




np.random.seed(42)

m, l, p, n = 6, 10, 5, 5
block_size  = 5
num_its     = 10000

A      = np.random.randn(m, l, n)
X_true = np.random.randn(l, p, n)
B      = tprod(A, X_true)
X0     = np.zeros((l, p, n))




_, its_tRK  = tRK( A, B, X0, num_its)
_, its_tRKB = tRKB(A, B, X0, num_its, block_size)




iters    = np.arange(num_its + 1)
res_tRK  = [np.linalg.norm(tprod(A, X) - B) for X in its_tRK]
res_tRKB = [np.linalg.norm(tprod(A, X) - B) for X in its_tRKB]

plt.figure(figsize=(8, 5))
plt.semilogy(iters, res_tRK,  'b-',  linewidth=2, label='tRK')
plt.semilogy(iters, res_tRKB, 'r--', linewidth=2, label=f'tRKB (block={block_size})')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('tRK_vs_tRKB.png', dpi=150, bbox_inches='tight')
plt.show()