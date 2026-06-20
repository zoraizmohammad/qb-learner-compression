"""
check_ibm_connection.py

One-command sanity check before spending QPU time: confirm the IBM Quantum account
is reachable, list the operational real devices with their queue depths, and report the
least-busy device. Reads credentials from the saved account (~/.qiskit/qiskit-ibm.json)
or the QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE environment variables. No token is stored
in this file. See docs/HARDWARE.md to set up an account.
"""
import os
import sys


def main():
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as e:
        print("qiskit-ibm-runtime is not installed:", e)
        sys.exit(1)

    try:
        # Prefer a saved account; fall back to env-var token if present.
        token = os.environ.get("QISKIT_IBM_TOKEN")
        instance = os.environ.get("QISKIT_IBM_INSTANCE")
        if token:
            service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                           token=token, instance=instance)
        else:
            service = QiskitRuntimeService()
    except Exception as e:
        print("Could not initialize QiskitRuntimeService:", e)
        print("Set up your account first (see docs/HARDWARE.md, section 2).")
        sys.exit(1)

    try:
        backends = service.backends(operational=True, simulator=False)
    except Exception as e:
        print("Connected, but could not list backends:", e)
        sys.exit(1)

    if not backends:
        print("Connected, but no operational real-device backends are available to this account.")
        sys.exit(0)

    print(f"Connected. {len(backends)} operational real device(s):\n")
    print(f"{'backend':<24}{'qubits':>7}{'pending jobs':>14}")
    print("-" * 45)
    rows = []
    for b in backends:
        try:
            st = b.status()
            nq = b.num_qubits
            pend = st.pending_jobs
        except Exception:
            nq, pend = "?", "?"
        rows.append((b.name, nq, pend))
        print(f"{b.name:<24}{str(nq):>7}{str(pend):>14}")

    try:
        lb = service.least_busy(operational=True, simulator=False)
        print(f"\nLeast busy: {lb.name}  ({lb.num_qubits} qubits, "
              f"{lb.status().pending_jobs} pending jobs)")
    except Exception as e:
        print("\n(could not determine least-busy backend:", e, ")")


if __name__ == "__main__":
    main()
