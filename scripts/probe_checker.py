"""
Probe: checkerboard categorization with a FREQUENCY knob (= difficulty).
label = (floor(f*x0) + floor(f*x1)) mod 2.  Higher f => finer cells => needs more
circuit capacity. Hypothesis: accuracy rises incrementally with active entanglers
=> a smooth, graded accuracy-vs-CNOT frontier with a visible knee.
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


def checker(freq, n=200, noise=0.0, seed=30):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    cell = np.floor(freq * X).sum(axis=1).astype(int)
    y = cell % 2
    if noise > 0:
        flip = rng.uniform(size=n) < noise
        y = np.where(flip, 1 - y, y)
    return X, y


def mask_with_k(k):
    flat = np.zeros(N_TERMS, dtype=int); flat[:k] = 1
    return flat.reshape(DEPTH, len(PAIRS), 3)


def train_one(X, y, mask, seed, n_iters=600, lr=0.05):
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
    for freq in (2, 3, 4):
        X, y = checker(freq)
        print(f"\n=== checkerboard freq={freq}  n={len(X)} balance={y.mean():.2f} ===")
        print(f"{'terms':>5} {'N2q':>4} {'mean_acc':>9} {'std':>6}")
        for k in range(N_TERMS, -1, -2):
            mask = mask_with_k(k)
            n2q = transpiled_2q_cost(NQ, DEPTH, PAIRS, mask, "heisenberg")
            accs = [train_one(X, y, mask, s) for s in seeds]
            print(f"{k:>5} {n2q:>4} {np.mean(accs):>9.3f} {np.std(accs):>6.3f}")


if __name__ == "__main__":
    main()
