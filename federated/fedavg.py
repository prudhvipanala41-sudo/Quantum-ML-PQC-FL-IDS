import torch

from fl_model import FLIDSModel
from local_train import train_client


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


def validate_state(state):

    for name, parameter in state.items():

        if torch.isnan(parameter).any():
            raise ValueError(f"NaN detected in {name}")

        if torch.isinf(parameter).any():
            raise ValueError(f"Infinity detected in {name}")


if __name__ == "__main__":

    rounds = 3

    global_model = FLIDSModel()

    global_state = {
        name: parameter.detach().clone()
        for name, parameter in global_model.state_dict().items()
    }

    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in global_state.items()
    }

    for round_number in range(1, rounds + 1):

        print("\n==============================")
        print(f"FEDERATED ROUND {round_number}")
        print("==============================")

        client_updates = []
        client_samples = []

        for client_id in range(1, 6):

            update, samples, loss = train_client(
                client_id=client_id,
                global_state=global_state,
                epochs=2
            )

            client_updates.append(update)
            client_samples.append(samples)

            print(
                f"Client {client_id} | "
                f"Samples: {samples} | "
                f"Loss: {loss:.6f}"
            )

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
            "Global model did not change after training."
        )

    print("\n==============================")
    print("MULTI-ROUND FL COMPLETED")
    print("==============================")

    print("Rounds:", rounds)
    print("Total samples:", sum(client_samples))
    print("Global model changed: YES")
    print("NaN validation: PASSED")
    print("Parameter compatibility: PASSED")

    print("\nFinal global model:")

    for name, parameter in global_model.state_dict().items():
        print(name, parameter.shape)
    import os

os.makedirs("results/federated", exist_ok=True)

torch.save(
    global_model.state_dict(),
    "results/federated/global_model.pt"
)

print("\nGlobal model saved to:")
print("results/federated/global_model.pt")