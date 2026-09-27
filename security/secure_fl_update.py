import io
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from secure_update import (
    establish_session_key,
    encrypt_model_update,
    decrypt_model_update
)

from federated.fl_model import FLIDSModel


def serialize_model_parameters(model):
    buffer = io.BytesIO()

    torch.save(
        model.state_dict(),
        buffer
    )

    return buffer.getvalue()


def deserialize_model_parameters(model, data):
    buffer = io.BytesIO(data)

    state_dict = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    return model


def secure_model_update(model):
    original_data = serialize_model_parameters(model)

    session_key, kem_ciphertext = establish_session_key()

    encrypted_update = encrypt_model_update(
        original_data,
        session_key,
        associated_data=b"FL-MODEL-UPDATE"
    )

    decrypted_data = decrypt_model_update(
        encrypted_update,
        session_key
    )

    data_integrity = decrypted_data == original_data

    return {
        "original_size": len(original_data),
        "encrypted_size": len(encrypted_update["ciphertext"]),
        "kem_ciphertext_size": len(kem_ciphertext),
        "nonce_size": len(encrypted_update["nonce"]),
        "session_key_size": len(session_key),
        "data_integrity": data_integrity,
        "encrypted_update": encrypted_update,
        "kem_ciphertext": kem_ciphertext
    }


if __name__ == "__main__":
    print("=== Secure Actual FL Model Update Test ===")

    model = FLIDSModel()

    result = secure_model_update(model)

    print("Original model update size:",
          result["original_size"], "bytes")

    print("ML-KEM ciphertext size:",
          result["kem_ciphertext_size"], "bytes")

    print("AES session key size:",
          result["session_key_size"], "bytes")

    print("AES-GCM nonce size:",
          result["nonce_size"], "bytes")

    print("Encrypted model update size:",
          result["encrypted_size"], "bytes")

    print("Model update integrity:",
          result["data_integrity"])

    print(
        "Secure actual FL model update PASSED:",
        result["data_integrity"]
    )