
import io
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.fl_model import FLIDSModel
from federated.local_train import train_client

from secure_update import (
    establish_session_key,
    encrypt_model_update,
    decrypt_model_update
)


NUM_CLIENTS = 5
NUM_ROUNDS = 3


def serialize_state_dict(state_dict):
    buffer = io.BytesIO()

    torch.save(
        state_dict,
        buffer
    )

    return buffer.getvalue()


def deserialize_state_dict(data):
    buffer = io.BytesIO(data)

    return torch.load(
        buffer,
        map_location="cpu",
        weights_only=True
    )


def secure_client_update(client_id, global_state):
    client_update, samples, loss = train_client(
        client_id=client_id,
        global_state=global_state,
        epochs=2,
        batch_size=128,
        learning_rate=0.001
    )

    original_data = serialize_state_dict(client_update)

    session_key, kem_ciphertext = establish_session_key()

    associated_data = f"FL-CLIENT-{client_id}".encode()

    encrypted_update = encrypt_model_update(
        original_data,
        session_key,
        associated_data=associated_data
    )

    server_data = decrypt_model_update(
        encrypted_update,
        session_key
    )

    integrity = server_data == original_data

    recovered_update = deserialize_state_dict(server_data)

    return (
        recovered_update,
        samples,
        loss,
        {
            "client_id": client_id,
            "original_size": len(original_data),
            "kem_ciphertext_size": len(kem_ciphertext),
            "encrypted_size": len(encrypted_update["ciphertext"]),
            "nonce_size": len(encrypted_update["nonce"]),
            "integrity": integrity
        }
    )


def secure_fedavg(client_updates):
    total_samples = sum(
        samples
        for _, samples, _ in client_updates
    )

    global_state = {}

    parameter_names = client_updates[0][0].keys()

    for name in parameter_names:

        weighted_parameter = torch.zeros_like(
            client_updates[0][0][name],
            dtype=torch.float32
        )

        for state_dict, samples, _ in client_updates:

            weighted_parameter += (
                state_dict[name].float()
                * (samples / total_samples)
            )

        global_state[name] = weighted_parameter

    return global_state


def states_differ(state_a, state_b):
    for name in state_a:
        if not torch.equal(state_a[name], state_b[name]):
            return True

    return False


if __name__ == "__main__":

    print("=== Secure Federated Learning with ML-KEM + AES-GCM ===")
    print()

    global_model = FLIDSModel()
    global_state = global_model.state_dict()

    initial_global_state = {
        name: parameter.detach().clone()
        for name, parameter in global_state.items()
    }

    all_rounds_passed = True

    for round_number in range(1, NUM_ROUNDS + 1):

        print(f"--- Secure FL Round {round_number} ---")

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

            print(
                f"Client {client_id}: "
                f"samples={samples}, "
                f"loss={loss:.6f}, "
                f"original={security_info['original_size']} bytes, "
                f"KEM={security_info['kem_ciphertext_size']} bytes, "
                f"encrypted={security_info['encrypted_size']} bytes, "
                f"integrity={security_info['integrity']}"
            )

            if not security_info["integrity"]:
                all_rounds_passed = False

        global_state = secure_fedavg(
            client_updates
        )

        print(
            f"Round {round_number} secure aggregation completed."
        )
        print()

    final_global_model = FLIDSModel()
    final_global_model.load_state_dict(global_state)

    global_model_changed = states_differ(
        initial_global_state,
        global_state
    )

    print("=== Secure FL Validation ===")
    print(
        "Rounds completed:",
        NUM_ROUNDS
    )
    print(
        "Clients per round:",
        NUM_CLIENTS
    )
    print(
        "All encrypted updates verified:",
        all_rounds_passed
    )
    print(
        "Global model changed:",
        global_model_changed
    )
    print(
        "Secure Federated Learning PASSED:",
        all_rounds_passed and global_model_changed
    )