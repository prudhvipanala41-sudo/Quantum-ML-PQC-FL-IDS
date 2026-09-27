import json
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os


project_root = Path(__file__).resolve().parent.parent
result_dir = project_root / "results" / "aes_gcm"
result_dir.mkdir(parents=True, exist_ok=True)

key = AESGCM.generate_key(bit_length=256)
aesgcm = AESGCM(key)

nonce = os.urandom(12)

original_data = b"Federated learning model update: client_1_weights"
associated_data = b"FL-Client-1"

ciphertext = aesgcm.encrypt(
    nonce,
    original_data,
    associated_data
)

decrypted_data = aesgcm.decrypt(
    nonce,
    ciphertext,
    associated_data
)

tampered_ciphertext = bytearray(ciphertext)
tampered_ciphertext[0] ^= 1

tampering_rejected = False

try:
    aesgcm.decrypt(
        nonce,
        bytes(tampered_ciphertext),
        associated_data
    )
except Exception:
    tampering_rejected = True

validation = {
    "encryption_succeeded": True,
    "decryption_succeeded": True,
    "original_data_recovered": decrypted_data == original_data,
    "authentication_tampering_rejected": tampering_rejected,
    "nonce_size_bytes": len(nonce),
    "key_size_bits": 256,
    "ciphertext_size_bytes": len(ciphertext)
}

passed = all([
    validation["encryption_succeeded"],
    validation["decryption_succeeded"],
    validation["original_data_recovered"],
    validation["authentication_tampering_rejected"]
])

artifact = {
    "algorithm": "AES-256-GCM",
    "test_type": "standalone",
    "validation": validation,
    "passed": passed
}

output_file = result_dir / "aes_gcm_validation.json"
output_file.write_text(
    json.dumps(artifact, indent=4),
    encoding="utf-8"
)

print("=== AES-256-GCM Validation Artifact ===")
print("Artifact:", output_file)
print("Artifact exists:", output_file.exists())
print("Algorithm:", artifact["algorithm"])
print("Key size:", validation["key_size_bits"], "bits")
print("Nonce size:", validation["nonce_size_bytes"], "bytes")
print("Encryption succeeded:", validation["encryption_succeeded"])
print("Decryption succeeded:", validation["decryption_succeeded"])
print("Original data recovered:", validation["original_data_recovered"])
print("Tampered ciphertext rejected:", validation["authentication_tampering_rejected"])
print("AES-256-GCM validation:", passed)