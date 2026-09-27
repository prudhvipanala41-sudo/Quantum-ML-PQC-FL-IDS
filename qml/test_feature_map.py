import pennylane as qml
import numpy as np


# ============================================================
# 1. QUANTUM DEVICE
# ============================================================

n_qubits = 8

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


# ============================================================
# 2. QUANTUM FEATURE MAP
# ============================================================

@qml.qnode(dev)
def feature_map(features):

    # Encode each classical feature into one qubit
    for i in range(n_qubits):
        qml.RY(features[i], wires=i)

    # Create interactions between neighboring qubits
    for i in range(n_qubits - 1):
        qml.CNOT(
            wires=[i, i + 1]
        )

    return qml.probs(
        wires=range(n_qubits)
    )


# ============================================================
# 3. TEST INPUT
# ============================================================

sample = np.array([
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8
])


# ============================================================
# 4. RUN QUANTUM CIRCUIT
# ============================================================

result = feature_map(sample)


# ============================================================
# 5. DISPLAY RESULT
# ============================================================

print("=" * 60)
print("QUANTUM FEATURE MAP TEST")
print("=" * 60)

print("\nNumber of qubits:", n_qubits)

print("\nInput features:")
print(sample)

print("\nNumber of quantum probabilities:")
print(len(result))

print("\nFirst 10 probabilities:")
print(result[:10])

print("\nSum of probabilities:")
print(np.sum(result))

print("\nQuantum feature map executed successfully.")