import pennylane as qml

n_qubits = 2

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def quantum_circuit():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    return qml.probs(wires=[0, 1])


result = quantum_circuit()

print("Quantum circuit executed successfully.")
print("Measurement probabilities:")
print(result)