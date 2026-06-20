"""Probe whether a 3-qubit / 3-feature parity task gives a GRADED accuracy-vs-CNOT frontier."""
import sys, os
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import qcore
from src.data import get_pothos_chater_parity
from src.hardware_cost import transpiled_2q_cost

NQ = 3
DEPTH = 3
PAIRS = [(0, 1), (1, 2)]            # linear chain
N_TERMS = DEPTH * len(PAIRS) * 3   # total maskable Heisenberg terms = 18
READOUT = ("Z0", "Z1", "Z2", "X0", "X1", "X2")


def mask_with_k(k):
    flat = np.zeros(N_TERMS, dtype=int)
    flat[:k] = 1
    return flat.reshape(DEPTH, len(PAIRS), 3)


def train_one(X, y, mask, seed, n_iters=400, lr=0.05):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    cfg = qcore.ModelConfig(n_qubits=NQ, depth=DEPTH, pairs=PAIRS, n_reupload=1, readout_paulis=READOUT)
    params = qcore.init_params(cfg, seed=seed)
    _, vg, predict = qcore.make_loss_and_grad(cfg, mask)
    opt = qcore.Adam(params, lr=lr)
    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr.astype(float))
    for _ in range(n_iters):
        _, g = vg(params, Xtr_j, ytr_j)
        params = opt.step(params, g)
    return qcore.accuracy(predict, params, jnp.asarray(Xte), yte)


def main(level="medium", seeds=(0, 1, 2)):
    params = dict(easy=dict(separation=0.6, std=0.08),
                  medium=dict(separation=0.5, std=0.12),
                  hard=dict(separation=0.42, std=0.16))[level]
    X, y = get_pothos_chater_parity(n_features=3, n_per_corner=10, seed=30, **params)
    print(f"3-qubit parity, difficulty={level} {params}, n={len(X)}")
    print(f"{'terms':>5} {'N2q':>4} {'mean_acc':>9} {'std':>6}")
    for k in range(N_TERMS, -1, -2):
        mask = mask_with_k(k)
        n2q = transpiled_2q_cost(NQ, DEPTH, PAIRS, mask, "heisenberg")
        accs = [train_one(X, y, mask, s) for s in seeds]
        print(f"{k:>5} {n2q:>4} {np.mean(accs):>9.3f} {np.std(accs):>6.3f}")


if __name__ == "__main__":
    main()
