import json
import sys
import io
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.fl_model import FLIDSModel
from secure_update import establish_session_key, encrypt_model_update


def serialize_model_update(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def main():
    print("=== Model / Communication Overhead Benchmark ===")
    print()

    model = FLIDSModel()

    model_update = serialize_model_update(model)
    plain_size = len(model_update)

    print(f"Plain model update size: {plain_size} bytes")

    client_shared_secret, kem_ciphertext = establish_session_key()

    encrypted_update = encrypt_model_update(
        model_update,
        client_shared_secret
    )

    kem_size = len(kem_ciphertext)

    encrypted_size = (
        len(encrypted_update["ciphertext"])
        + len(encrypted_update["nonce"])
        + len(encrypted_update["associated_data"])
    )

    secure_payload_size = kem_size + encrypted_size
    additional_bytes = secure_payload_size - plain_size
    overhead_percentage = (additional_bytes / plain_size) * 100

    print(f"ML-KEM ciphertext size: {kem_size} bytes")
    print(f"AES-GCM encrypted update size: {encrypted_size} bytes")
    print(f"Secure communication payload: {secure_payload_size} bytes")
    print(f"Additional communication: {additional_bytes} bytes")
    print(f"Communication overhead: {overhead_percentage:.2f}%")

    result = {
        "benchmark": "Model Communication Overhead",
        "model": "FLIDSModel",
        "plain_model_update_bytes": plain_size,
        "ml_kem_ciphertext_bytes": kem_size,
        "aes_gcm_encrypted_update_bytes": encrypted_size,
        "secure_communication_payload_bytes": secure_payload_size,
        "additional_communication_bytes": additional_bytes,
        "communication_overhead_percentage": overhead_percentage
    }

    output_dir = PROJECT_ROOT / "results" / "benchmarking"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model_communication_overhead.json"

    with open(output_path, "w") as file:
        json.dump(result, file, indent=4)

    print()
    print("Benchmark artifact saved:")
    print(output_path)


if __name__ == "__main__":
    main()