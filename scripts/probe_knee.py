"""
probe_knee.py

Empirical check that the accuracy-vs-CNOT frontier has a real knee on the
XOR-style Pothos-Chater task, across difficulty levels.

For each difficulty and each number of active Heisenberg entanglers k in {0,1,2,3}
(=> FakeManila N2q in {0,6,12,18}), train the qcore learner with SINGLE-QUBIT-ONLY
readout (so feature interaction must come from entanglement) and report mean test
accuracy over seeds. A knee = accuracy stays high for large k, then drops sharply.
"""
import sys, os
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import qcore
from src.data import get_difficulty_dataset, DIFFICULTY_LEVELS
from src.hardware_cost import transpiled_2q_cost


def mask_with_k(depth, k):
    """Activate k of the depth*3 individual Heisenberg interaction terms."""
    flat = np.zeros(depth * 3, dtype=int)
    flat[:k] = 1
    return flat.reshape(depth, 1, 3)


def train_one(X, y, mask, seed, n_iters=300, lr=0.05):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    cfg = qcore.ModelConfig(
        n_qubits=2, depth=3, pairs=[(0, 1)], n_reupload=1,
        readout_paulis=("Z0", "Z1", "X0", "X1"),  # single-qubit only -> entanglement matters
    )
    params = qcore.init_params(cfg, seed=seed)
    loss_fn, vg, predict = qcore.make_loss_and_grad(cfg, mask)
    opt = qcore.Adam(params, lr=lr)
    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr.astype(float))
    for _ in range(n_iters):
        _, g = vg(params, Xtr_j, ytr_j)
        params = opt.step(params, g)
    return qcore.accuracy(predict, params, jnp.asarray(Xte), yte)


def main(seeds=(0, 1, 2)):
    depth = 3
    for level in DIFFICULTY_LEVELS:
        X, y = get_difficulty_dataset(level, n_per_cluster=15, seed=30)
        print(f"\n=== difficulty={level}  ({DIFFICULTY_LEVELS[level]})  n={len(X)} ===")
        print(f"{'terms':>5} {'N2q':>4} {'mean_acc':>9} {'std':>6}")
        for k in range(9, -1, -1):
            mask = mask_with_k(depth, k)
            n2q = transpiled_2q_cost(2, depth, [(0, 1)], mask, "heisenberg")
            accs = [train_one(X, y, mask, s) for s in seeds]
            print(f"{k:>5} {n2q:>4} {np.mean(accs):>9.3f} {np.std(accs):>6.3f}")


if __name__ == "__main__":
    main()
