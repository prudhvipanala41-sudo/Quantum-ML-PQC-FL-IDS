
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.fl_model import FLIDSModel
from federated.local_train import train_client

from secure_fedavg import secure_client_update, secure_fedavg


NUM_CLIENTS = 5
NUM_ROUNDS = 3


def run_plain_fl():
    model = FLIDSModel()

    global_state = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }

    start = time.perf_counter()

    for _ in range(NUM_ROUNDS):

        client_updates = []

        for client_id in range(1, NUM_CLIENTS + 1):

            update, samples, loss = train_client(
                client_id=client_id,
                global_state=global_state,
                epochs=2,
                batch_size=128,
                learning_rate=0.001
            )

            client_updates.append(
                (
                    update,
                    samples,
                    loss
                )
            )

        total_samples = sum(
            samples
            for _, samples, _ in client_updates
        )

        new_global_state = {}

        for name in global_state:

            weighted_parameter = torch.zeros_like(
                global_state[name],
                dtype=torch.float32
            )

            for update, samples, _ in client_updates:

                weighted_parameter += (
                    update[name].float()
                    * (samples / total_samples)
                )

            new_global_state[name] = weighted_parameter

        global_state = new_global_state

    elapsed = time.perf_counter() - start

    return elapsed


def run_secure_fl():
    model = FLIDSModel()

    global_state = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }

    start = time.perf_counter()

    for _ in range(NUM_ROUNDS):

        client_updates = []

        for client_id in range(1, NUM_CLIENTS + 1):

            (
                recovered_update,
                samples,
                loss,
                security_info
            ) = secure_client_update(
                client_id,
                global_state
            )

            client_updates.append(
                (
                    recovered_update,
                    samples,
                    loss
                )
            )

        global_state = secure_fedavg(
            client_updates
        )

    elapsed = time.perf_counter() - start

    return elapsed


if __name__ == "__main__":

    print("=== Plain FL vs Secure FL Runtime Benchmark ===")
    print()

    print("Running plain FL...")
    plain_runtime = run_plain_fl()

    print(
        f"Plain FL runtime: "
        f"{plain_runtime:.4f} seconds"
    )

    print()

    print("Running Secure FL...")
    secure_runtime = run_secure_fl()

    print(
        f"Secure FL runtime: "
        f"{secure_runtime:.4f} seconds"
    )

    overhead_seconds = (
        secure_runtime - plain_runtime
    )

    overhead_percentage = (
        overhead_seconds
        / plain_runtime
        * 100
    )

    result = {
        "benchmark": "Plain FL vs Secure FL Runtime",
        "clients_per_round": NUM_CLIENTS,
        "rounds": NUM_ROUNDS,
        "plain_fl_runtime_seconds": plain_runtime,
        "secure_fl_runtime_seconds": secure_runtime,
        "additional_runtime_seconds": overhead_seconds,
        "runtime_overhead_percentage": overhead_percentage
    }

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "benchmarking"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "plain_vs_secure_fl_runtime.json"
    )

    with open(output_path, "w") as file:
        json.dump(
            result,
            file,
            indent=4
        )

    print()
    print("=== Runtime Comparison ===")
    print(
        f"Plain FL:   "
        f"{plain_runtime:.4f} seconds"
    )
    print(
        f"Secure FL:  "
        f"{secure_runtime:.4f} seconds"
    )
    print(
        f"Additional: "
        f"{overhead_seconds:.4f} seconds"
    )
    print(
        f"Overhead:   "
        f"{overhead_percentage:.2f}%"
    )

    print()
    print("Benchmark artifact saved:")
    print(output_path)