"""
run_on_hardware.py

Deploy the trained quantum Bayesian learner on real (or realistically-noisy) hardware
and compare against exact simulation. Training is done in simulation; only inference
runs on the chosen backend.

Examples
--------
# Free, no account: validate under the FakeManila noise model
python scripts/run_on_hardware.py --difficulty hard --mode noisy --shots 4096

# Exact bridge check (should equal the simulator)
python scripts/run_on_hardware.py --difficulty hard --mode exact

# Real IBM hardware (needs a saved account / QISKIT_IBM_TOKEN; see docs/HARDWARE.md)
python scripts/run_on_hardware.py --difficulty hard --mode ibm --n-eval 20 --shots 4096

# Train once, save, and reuse the saved model later
python scripts/run_on_hardware.py --difficulty hard --mode noisy --save results_v2/hardware/model.npz
python scripts/run_on_hardware.py --load results_v2/hardware/model.npz --mode ibm

It evaluates BOTH the full circuit and a greedy-compressed circuit, so you can see the
two-qubit cost drop (cheaper on hardware) while accuracy is retained.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import qcore, hardware_backend as hw
from src.data import get_checker_difficulty
from src.hardware_cost import transpiled_2q_cost
from src.run_v2 import _cfg, greedy_trajectory, Engine, N_QUBITS, DEPTH, PAIRS, ENTANGLER


def _n2q(mask):
    return transpiled_2q_cost(N_QUBITS, DEPTH, PAIRS, np.asarray(mask, dtype=int), ENTANGLER)


def _train_params_for_mask(cfg, Xtr, ytr, mask, seed):
    eng = Engine(cfg)
    return eng.train(Xtr, ytr, mask, seed, 600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    ap.add_argument("--mode", default="noisy", choices=["exact", "noisy", "ibm"])
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=160, help="dataset size")
    ap.add_argument("--n-eval", type=int, default=None,
                    help="evaluate on at most this many test points (caps QPU cost for --mode ibm)")
    ap.add_argument("--target-n2q", type=int, default=6, help="compressed-circuit target cost")
    ap.add_argument("--ibm-backend", default=None, help="specific IBM backend name (else least busy)")
    ap.add_argument("--save", default=None, help="save the trained full model to this .npz")
    ap.add_argument("--load", default=None, help="load a saved .npz model instead of training")
    args = ap.parse_args()

    if args.load:
        params, mask, cfg = hw.load_model(args.load)
        X, y = get_checker_difficulty(args.difficulty, n=args.n, seed=30)
        _, Xte, _, yte = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
        models = [("loaded", params, mask)]
    else:
        cfg = _cfg()
        X, y = get_checker_difficulty(args.difficulty, n=args.n, seed=30)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
        eng = Engine(cfg)
        full_mask = qcore.full_mask(cfg)
        print("training full model in simulation ...")
        full_params = eng.train(Xtr, ytr, full_mask, args.seed, 600)
        # greedy-compressed model near target_n2q
        print("building greedy-compressed model ...")
        traj = greedy_trajectory(eng, Xtr, ytr, Xte, yte, args.seed)
        comp = min(traj, key=lambda p: abs(p["n2q"] - args.target_n2q))
        comp_params = _train_params_for_mask(cfg, Xtr, ytr, comp["mask"], args.seed)
        models = [("full", full_params, full_mask),
                  (f"compressed~{args.target_n2q}", comp_params, comp["mask"])]
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            hw.save_model(args.save, full_params, full_mask, cfg)
            print(f"saved full model -> {args.save}")

    if args.n_eval:
        Xte, yte = Xte[:args.n_eval], yte[:args.n_eval]

    print(f"\nBackend mode: {args.mode}  | shots: {args.shots}  | "
          f"eval points: {len(Xte)}  | difficulty: {args.difficulty}\n")
    try:
        estimator, target = hw.make_estimator(args.mode, shots=args.shots,
                                               ibm_backend=args.ibm_backend)
    except Exception as e:
        print(f"could not initialize '{args.mode}' backend: {e}")
        if args.mode == "ibm":
            print("See docs/HARDWARE.md to set up your IBM Quantum account/token.")
        return

    # exact reference (always cheap) for comparison
    est_exact, _ = hw.make_estimator("exact")

    print(f"{'model':<16}{'N2q':>5}{'acc(exact)':>12}{'acc('+args.mode+')':>14}")
    print("-" * 47)
    for name, params, mask in models:
        n2q = _n2q(mask)
        acc_ref = hw.evaluate(params, mask, Xte, yte, cfg, est_exact)["accuracy"]
        acc_dev = hw.evaluate(params, mask, Xte, yte, cfg, estimator,
                              transpile_target=target)["accuracy"]
        print(f"{name:<16}{n2q:>5}{acc_ref:>12.3f}{acc_dev:>14.3f}")
    print("\ndone.")


if __name__ == "__main__":
    main()
