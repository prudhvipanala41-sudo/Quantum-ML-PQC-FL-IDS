from pqcrypto.kem import ml_kem_768
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import json
import os
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
result_dir = project_root / "results" / "ml_kem_aes"
result_dir.mkdir(parents=True, exist_ok=True)

# ML-KEM-768
public_key, secret_key = ml_kem_768.keygen()

kem_ciphertext, client_shared_secret = ml_kem_768.encaps(public_key)

server_shared_secret = ml_kem_768.decaps(
    secret_key,
    kem_ciphertext
)

ml_kem_shared_secret_match = (
    client_shared_secret == server_shared_secret
)

# AES-256-GCM using ML-KEM shared secret
aesgcm = AESGCM(client_shared_secret)

model_update = b"Federated client model update: weights_and_parameters"
associated_data = b"FL-Client-1"
nonce = os.urandom(12)

encrypted_update = aesgcm.encrypt(
    nonce,
    model_update,
    associated_data
)

decrypted_update = aesgcm.decrypt(
    nonce,
    encrypted_update,
    associated_data
)

aes_decryption_match = (
    decrypted_update == model_update
)

validation = {
    "ml_kem_algorithm": "MlKem768",
    "ml_kem_shared_secret_size_bytes": len(client_shared_secret),
    "ml_kem_shared_secret_match": ml_kem_shared_secret_match,
    "aes_algorithm": "AES-256-GCM",
    "aes_key_size_bits": len(client_shared_secret) * 8,
    "nonce_size_bytes": len(nonce),
    "encryption_succeeded": True,
    "decryption_succeeded": True,
    "original_model_update_recovered": aes_decryption_match
}

passed = all([
    validation["ml_kem_shared_secret_match"],
    validation["encryption_succeeded"],
    validation["decryption_succeeded"],
    validation["original_model_update_recovered"]
])

artifact = {
    "test_type": "ML-KEM-768 + AES-256-GCM integration",
    "validation": validation,
    "passed": passed
}

output_file = result_dir / "ml_kem_aes_validation.json"

output_file.write_text(
    json.dumps(artifact, indent=4),
    encoding="utf-8"
)

print("=== ML-KEM-768 + AES-256-GCM Validation Artifact ===")
print("Artifact:", output_file)
print("Artifact exists:", output_file.exists())
print("ML-KEM shared secret size:", len(client_shared_secret), "bytes")
print("ML-KEM shared secrets match:", ml_kem_shared_secret_match)
print("AES key size:", len(client_shared_secret) * 8, "bits")
print("AES encryption succeeded:", True)
print("AES decryption succeeded:", True)
print("Original model update recovered:", aes_decryption_match)
print("Integration validation:", passed)