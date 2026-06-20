"""
Probe: does a radial/concentric P&C similarity task give a GRADED (smooth)
accuracy-vs-CNOT frontier on 2 qubits? Category A = near prototype (typical),
Category B = far from prototype (atypical). Boundary is a circle => curved =>
low-capacity circuits underfit (partial accuracy), more entanglers fit it better.
"""
import sys, os
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import qcore
from src.hardware_cost import transpiled_2q_cost

NQ, DEPTH, PAIRS = 2, 4, [(0, 1)]
N_TERMS = DEPTH * len(PAIRS) * 3
READOUT = ("Z0", "Z1", "X0", "X1")


def concentric(n=160, r_split=0.30, noise=0.04, seed=30):
    rng = np.random.default_rng(seed)
    c = np.array([0.5, 0.5])
    X, y = [], []
    for _ in range(n):
        # uniform in the square, label by radius from center (typical vs atypical)
        p = rng.uniform(0.05, 0.95, size=2)
        r = np.linalg.norm(p - c)
        lab = 0 if r < r_split else 1
        # soft boundary: flip near the boundary by gaussian noise on r
        if abs(r - r_split) < noise:
            lab = int(rng.integers(0, 2))
        X.append(p); y.append(lab)
    return np.array(X), np.array(y)


def mask_with_k(k):
    flat = np.zeros(N_TERMS, dtype=int); flat[:k] = 1
    return flat.reshape(DEPTH, len(PAIRS), 3)


def train_one(X, y, mask, seed, n_iters=500, lr=0.05):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    cfg = qcore.ModelConfig(n_qubits=NQ, depth=DEPTH, pairs=PAIRS, n_reupload=2, readout_paulis=READOUT)
    params = qcore.init_params(cfg, seed=seed)
    _, vg, predict = qcore.make_loss_and_grad(cfg, mask)
    opt = qcore.Adam(params, lr=lr)
    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr.astype(float))
    for _ in range(n_iters):
        _, g = vg(params, Xtr_j, ytr_j); params = opt.step(params, g)
    return qcore.accuracy(predict, params, jnp.asarray(Xte), yte)


def main(seeds=(0, 1, 2)):
    X, y = concentric()
    print(f"concentric task n={len(X)} class balance={y.mean():.2f}")
    print(f"{'terms':>5} {'N2q':>4} {'mean_acc':>9} {'std':>6}")
    for k in range(N_TERMS, -1, -1):
        mask = mask_with_k(k)
        n2q = transpiled_2q_cost(NQ, DEPTH, PAIRS, mask, "heisenberg")
        accs = [train_one(X, y, mask, s) for s in seeds]
        print(f"{k:>5} {n2q:>4} {np.mean(accs):>9.3f} {np.std(accs):>6.3f}")


if __name__ == "__main__":
    main()
