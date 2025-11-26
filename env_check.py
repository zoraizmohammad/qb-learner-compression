# ============================
# env_check.py
# ============================

print("=== Environment Check Starting ===")

# 1. Basic imports
try:
    import numpy as np
    print("[OK] numpy imported")
except Exception as e:
    print("[ERROR] numpy:", e)

try:
    import qiskit
    from qiskit import QuantumCircuit
    print("[OK] qiskit imported")
except Exception as e:
    print("[ERROR] qiskit:", e)

try:
    import scipy
    print("[OK] scipy imported")
except Exception as e:
    print("[ERROR] scipy:", e)

# 2. Project imports
try:
    from src.data import get_toy_dataset
    print("[OK] project imports work")
except Exception as e:
    print("[ERROR] project import failed:", e)

# 3. Dataset check
try:
    X, y = get_toy_dataset()
    print("[OK] dataset loaded")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X sample:", X[:4])
    print("y sample:", y[:4])
except Exception as e:
    print("[ERROR] dataset loading failed:", e)

print("=== Environment Check Complete ===")

