import io
import json
import sys
import time
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


ITERATIONS = 20


def serialize_model(model):
    buffer = io.BytesIO()

    torch.save(
        model.state_dict(),
        buffer
    )

    return buffer.getvalue()


def benchmark():
    model = FLIDSModel()

    model_update = serialize_model(model)

    keygen_times = []
    encapsulation_times = []
    decapsulation_times = []
    encryption_times = []
    decryption_times = []

    successful_runs = 0

    for _ in range(ITERATIONS):

        start = time.perf_counter()

        session_key, kem_ciphertext = establish_session_key()

        keygen_and_encapsulation_time = (
            time.perf_counter() - start
        )

        keygen_times.append(
            keygen_and_encapsulation_time
        )

        start = time.perf_counter()

        encrypted_update = encrypt_model_update(
            model_update,
            session_key,
            associated_data=b"FL-BENCHMARK"
        )

        encryption_times.append(
            time.perf_counter() - start
        )

        start = time.perf_counter()

        decrypted_update = decrypt_model_update(
            encrypted_update,
            session_key
        )

        decryption_times.append(
            time.perf_counter() - start
        )

        if decrypted_update == model_update:
            successful_runs += 1

    result = {
        "stage": "Benchmarking",
        "benchmark": "Secure FL Cryptographic Overhead",
        "iterations": ITERATIONS,
        "successful_runs": successful_runs,
        "model_update_size_bytes": len(model_update),
        "kem_ciphertext_size_bytes": len(kem_ciphertext),
        "aes_nonce_size_bytes": len(encrypted_update["nonce"]),
        "encrypted_update_size_bytes": len(
            encrypted_update["ciphertext"]
        ),
        "keygen_encapsulation_mean_ms": (
            sum(keygen_times) / len(keygen_times) * 1000
        ),
        "encryption_mean_ms": (
            sum(encryption_times) / len(encryption_times) * 1000
        ),
        "decryption_mean_ms": (
            sum(decryption_times) / len(decryption_times) * 1000
        ),
        "total_crypto_mean_ms": (
            (
                sum(keygen_times)
                + sum(encryption_times)
                + sum(decryption_times)
            )
            / ITERATIONS
            * 1000
        ),
        "all_integrity_checks_passed": (
            successful_runs == ITERATIONS
        )
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
        / "secure_fl_crypto_benchmark.json"
    )

    with open(output_path, "w") as file:
        json.dump(
            result,
            file,
            indent=4
        )

    print("=== Secure FL Cryptographic Benchmark ===")
    print()
    print("Iterations:", result["iterations"])
    print("Successful runs:", result["successful_runs"])
    print(
        "Model update size:",
        result["model_update_size_bytes"],
        "bytes"
    )
    print(
        "ML-KEM ciphertext:",
        result["kem_ciphertext_size_bytes"],
        "bytes"
    )
    print(
        "AES-GCM nonce:",
        result["aes_nonce_size_bytes"],
        "bytes"
    )
    print(
        "Encrypted update:",
        result["encrypted_update_size_bytes"],
        "bytes"
    )
    print(
        "Key establishment mean:",
        f"{result['keygen_encapsulation_mean_ms']:.4f}",
        "ms"
    )
    print(
        "AES encryption mean:",
        f"{result['encryption_mean_ms']:.4f}",
        "ms"
    )
    print(
        "AES decryption mean:",
        f"{result['decryption_mean_ms']:.4f}",
        "ms"
    )
    print(
        "Total crypto mean:",
        f"{result['total_crypto_mean_ms']:.4f}",
        "ms"
    )
    print(
        "All integrity checks passed:",
        result["all_integrity_checks_passed"]
    )
    print()
    print("Benchmark artifact saved:")
    print(output_path)


if __name__ == "__main__":
    benchmark()