"""Quick P1 DoD check: does the diagnostic baseline (lam=0, full mask) learn?"""
import sys, os, time
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import qcore
from src.data import get_toy_dataset


def main(seed=42, n_iters=300, lr=0.05, depth=3, n_reupload=2):
    X, y = get_toy_dataset("pothos_chater_large")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    Xtr = jnp.asarray(Xtr); ytr_j = jnp.asarray(ytr.astype(float))
    Xte = jnp.asarray(Xte)

    cfg = qcore.ModelConfig(n_qubits=2, depth=depth, pairs=[(0, 1)], n_reupload=n_reupload)
    mask = qcore.full_mask(cfg)
    params = qcore.init_params(cfg, seed=seed)

    loss_fn, vg, predict = qcore.make_loss_and_grad(cfg, mask)
    opt = qcore.Adam(params, lr=lr)

    t0 = time.time()
    for it in range(n_iters):
        loss, grads = vg(params, Xtr, ytr_j)
        params = opt.step(params, grads)
        if it % 50 == 0 or it == n_iters - 1:
            tr_acc = qcore.accuracy(predict, params, Xtr, ytr)
            te_acc = qcore.accuracy(predict, params, Xte, yte)
            print(f"it {it:4d}  loss {float(loss):.4f}  train_acc {tr_acc:.3f}  test_acc {te_acc:.3f}")
    dt = time.time() - t0

    tr_acc = qcore.accuracy(predict, params, Xtr, ytr)
    te_acc = qcore.accuracy(predict, params, Xte, yte)
    final_ce = float(loss_fn(params, Xtr, ytr_j))
    print(f"\nFINAL: train_acc={tr_acc:.3f} test_acc={te_acc:.3f} train_CE={final_ce:.4f} "
          f"({n_iters} iters in {dt:.1f}s)")
    dod = (tr_acc >= 0.85 and te_acc >= 0.75)
    print("P1 DoD", "PASS" if dod else "FAIL", "(need train>=0.85, test>=0.75)")
    return dod


if __name__ == "__main__":
    main()
