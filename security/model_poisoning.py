import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.fl_model import FLIDSModel
from federated.local_train import train_client


POISONED_CLIENT_ID = 1
POISON_FACTOR = -3.0
ROUNDS = 3
CLIENT_COUNT = 5


def fedavg(client_updates, client_samples):

    total_samples = sum(client_samples)

    global_state = {}

    for parameter_name in client_updates[0]:

        weighted_parameter = torch.zeros_like(
            client_updates[0][parameter_name]
        )

        for update, samples in zip(client_updates, client_samples):

            weight = samples / total_samples

            weighted_parameter += (
                update[parameter_name] * weight
            )

        global_state[parameter_name] = weighted_parameter

    return global_state


def poison_update(client_update, poison_factor):

    poisoned_update = {}

    for parameter_name, parameter in client_update.items():

        poisoned_update[parameter_name] = (
            parameter.detach().clone() * poison_factor
        )

    return poisoned_update


def validate_state(state):

    for name, parameter in state.items():

        if torch.isnan(parameter).any():
            raise ValueError(f"NaN detected in {name}")

        if torch.isinf(parameter).any():
            raise ValueError(f"Infinity detected in {name}")


def main():

    print("========================================")
    print("MODEL POISONING ATTACK EXPERIMENT")
    print("========================================")

    print("Poisoned client:", POISONED_CLIENT_ID)
    print("Poison factor:", POISON_FACTOR)
    print("Rounds:", ROUNDS)
    print("Clients per round:", CLIENT_COUNT)

    global_model = FLIDSModel()

    global_state = {
        name: parameter.detach().clone()
        for name, parameter in global_model.state_dict().items()
    }

    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in global_state.items()
    }

    for round_number in range(1, ROUNDS + 1):

        print("\n==============================")
        print(f"POISONED FEDERATED ROUND {round_number}")
        print("==============================")

        client_updates = []
        client_samples = []

        for client_id in range(1, CLIENT_COUNT + 1):

            update, samples, loss = train_client(
                client_id=client_id,
                global_state=global_state,
                epochs=2
            )

            if client_id == POISONED_CLIENT_ID:

                update = poison_update(
                    update,
                    POISON_FACTOR
                )

                print(
                    f"Client {client_id} | "
                    f"POISONED | "
                    f"Samples: {samples} | "
                    f"Loss: {loss:.6f}"
                )

            else:

                print(
                    f"Client {client_id} | "
                    f"Clean | "
                    f"Samples: {samples} | "
                    f"Loss: {loss:.6f}"
                )

            client_updates.append(update)
            client_samples.append(samples)

        global_state = fedavg(
            client_updates,
            client_samples
        )

        validate_state(global_state)

        global_model.load_state_dict(global_state)

        print(
            f"Round {round_number} aggregation complete."
        )

    changed = False

    for name in initial_state:

        if not torch.equal(
            initial_state[name],
            global_state[name]
        ):
            changed = True
            break

    if not changed:
        raise ValueError(
            "Global model did not change after poisoned training."
        )

    output_dir = PROJECT_ROOT / "results" / "poisoning"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "poisoned_global_model.pt"

    torch.save(
        global_model.state_dict(),
        output_path
    )

    print("\n========================================")
    print("MODEL POISONING EXPERIMENT COMPLETED")
    print("========================================")

    print("Poisoned client:", POISONED_CLIENT_ID)
    print("Poison factor:", POISON_FACTOR)
    print("Rounds:", ROUNDS)
    print("Total samples:", sum(client_samples))
    print("Global model changed: YES")
    print("NaN validation: PASSED")
    print("Poisoned global model saved:")
    print(output_path)


if __name__ == "__main__":
    main()