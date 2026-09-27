import pennylane as qml
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

n_qubits = 8

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


# ============================================================
# 2. QUANTUM FEATURE MAP
# ============================================================

def feature_map(features):

    for i in range(n_qubits):
        qml.RY(
            features[i],
            wires=i
        )

    for i in range(n_qubits - 1):
        qml.CNOT(
            wires=[i, i + 1]
        )


# ============================================================
# 3. QUANTUM KERNEL
# ============================================================

@qml.qnode(dev)
def quantum_kernel_circuit(x1, x2):

    # Prepare first quantum state
    feature_map(x1)

    # Reverse preparation of second state
    for i in range(n_qubits - 2, -1, -1):
        qml.CNOT(
            wires=[i, i + 1]
        )

    for i in range(n_qubits - 1, -1, -1):
        qml.RY(
            -x2[i],
            wires=i
        )

    return qml.probs(wires=range(n_qubits))


def quantum_kernel(x1, x2):

    probabilities = quantum_kernel_circuit(
        x1,
        x2
    )

    # Probability of |00000000>
    return float(probabilities[0])


# ============================================================
# 4. TEST SAMPLES
# ============================================================

sample_a = np.array([
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8
])

sample_b = np.array([
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8
])

sample_c = np.array([
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
    0.1
])


# ============================================================
# 5. CALCULATE KERNEL VALUES
# ============================================================

kernel_same = quantum_kernel(
    sample_a,
    sample_b
)

kernel_different = quantum_kernel(
    sample_a,
    sample_c
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("QUANTUM KERNEL TEST")
print("=" * 60)

print("\nKernel(A, A):")
print(kernel_same)

print("\nKernel(A, B) where A and B are identical:")
print(kernel_same)

print("\nKernel(A, C) where C is different:")
print(kernel_different)

print("\nKernel values must be between 0 and 1.")

print("\nQuantum kernel executed successfully.")