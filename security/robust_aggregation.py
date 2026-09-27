import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.fl_model import FLIDSModel
from federated.local_train import train_client


ROUNDS = 3
CLIENT_COUNT = 5
POISONED_CLIENT_ID = 1
POISON_FACTOR = -3.0


def median_aggregate(client_updates):

    robust_state = {}

    for parameter_name in client_updates[0]:

        stacked_parameters = torch.stack(
            [
                update[parameter_name]
                for update in client_updates
            ]
        )

        robust_state[parameter_name] = torch.median(
            stacked_parameters,
            dim=0
        ).values

    return robust_state


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
    print("ROBUST MEDIAN AGGREGATION")
    print("========================================")

    print("Rounds:", ROUNDS)
    print("Clients per round:", CLIENT_COUNT)
    print("Poisoned client:", POISONED_CLIENT_ID)
    print("Poison factor:", POISON_FACTOR)

    global_model = FLIDSModel()

    global_state = {
        name: parameter.detach().clone()
        for name, parameter in global_model.state_dict().items()
    }

    for round_number in range(1, ROUNDS + 1):

        print("\n==============================")
        print(f"ROBUST FEDERATED ROUND {round_number}")
        print("==============================")

        client_updates = []

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
                    f"CLEAN | "
                    f"Samples: {samples} | "
                    f"Loss: {loss:.6f}"
                )

            client_updates.append(update)

        global_state = median_aggregate(
            client_updates
        )

        validate_state(global_state)

        global_model.load_state_dict(
            global_state
        )

        print(
            f"Round {round_number} "
            f"median aggregation complete."
        )

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "robust_aggregation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "robust_global_model.pt"
    )

    torch.save(
        global_model.state_dict(),
        output_path
    )

    print("\n========================================")
    print("ROBUST AGGREGATION COMPLETED")
    print("========================================")

    print("Aggregation method: Coordinate-wise Median")
    print("Rounds:", ROUNDS)
    print("Clients:", CLIENT_COUNT)
    print("Poisoned client:", POISONED_CLIENT_ID)
    print("NaN validation: PASSED")
    print("Robust model saved:")
    print(output_path)


if __name__ == "__main__":
    main()