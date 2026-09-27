
import io
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.fl_model import FLIDSModel

from secure_update import (
    establish_session_key,
    encrypt_model_update,
    decrypt_model_update
)


NUM_CLIENTS = 5


def serialize_model(model):
    buffer = io.BytesIO()

    torch.save(
        model.state_dict(),
        buffer
    )

    return buffer.getvalue()


def secure_client_update(model, client_id):
    original_data = serialize_model(model)

    session_key, kem_ciphertext = establish_session_key()

    encrypted_update = encrypt_model_update(
        original_data,
        session_key,
        associated_data=f"FL-CLIENT-{client_id}".encode()
    )

    decrypted_data = decrypt_model_update(
        encrypted_update,
        session_key
    )

    integrity = decrypted_data == original_data

    return {
        "client_id": client_id,
        "original_size": len(original_data),
        "kem_ciphertext_size": len(kem_ciphertext),
        "encrypted_size": len(encrypted_update["ciphertext"]),
        "nonce_size": len(encrypted_update["nonce"]),
        "integrity": integrity
    }


if __name__ == "__main__":
    print("=== Secure Federated Learning Transport Test ===")
    print()

    results = []

    for client_id in range(1, NUM_CLIENTS + 1):
        model = FLIDSModel()

        result = secure_client_update(
            model,
            client_id
        )

        results.append(result)

        print(
            f"Client {client_id}: "
            f"Original={result['original_size']} bytes, "
            f"KEM={result['kem_ciphertext_size']} bytes, "
            f"Encrypted={result['encrypted_size']} bytes, "
            f"Nonce={result['nonce_size']} bytes, "
            f"Integrity={result['integrity']}"
        )

    all_passed = all(
        result["integrity"]
        for result in results
    )

    print()
    print("Clients tested:", NUM_CLIENTS)
    print("All secure updates verified:", all_passed)
    print(
        "Secure FL transport PASSED:",
        all_passed
    )