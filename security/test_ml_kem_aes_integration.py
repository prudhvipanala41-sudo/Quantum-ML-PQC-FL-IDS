from pqcrypto.kem import ml_kem_768
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os


print("=== ML-KEM-768 + AES-256-GCM Integration Test ===")

# --------------------------------------------------
# 1. ML-KEM-768 key generation
# --------------------------------------------------

public_key, secret_key = ml_kem_768.keygen()

print("\nML-KEM key generation:")
print("Public key size:", len(public_key), "bytes")
print("Secret key size:", len(secret_key), "bytes")

# --------------------------------------------------
# 2. ML-KEM encapsulation
# --------------------------------------------------

ciphertext_kem, shared_secret_client = ml_kem_768.encaps(public_key)

print("\nML-KEM encapsulation:")
print("KEM ciphertext size:", len(ciphertext_kem), "bytes")
print("Shared secret size:", len(shared_secret_client), "bytes")

# --------------------------------------------------
# 3. ML-KEM decapsulation
# --------------------------------------------------

shared_secret_server = ml_kem_768.decaps(
    secret_key,
    ciphertext_kem
)

print("\nML-KEM decapsulation:")
print(
    "Shared secrets match:",
    shared_secret_client == shared_secret_server
)

# --------------------------------------------------
# 4. Use ML-KEM shared secret as AES key
# --------------------------------------------------

aes_key = shared_secret_client

print("\nAES key:")
print("AES key size:", len(aes_key) * 8, "bits")

aesgcm = AESGCM(aes_key)

# --------------------------------------------------
# 5. Simulated federated model update
# --------------------------------------------------

model_update = (
    b"Federated client model update: "
    b"weights_and_parameters"
)

associated_data = b"FL-Client-1"

nonce = os.urandom(12)

# --------------------------------------------------
# 6. AES-256-GCM encryption
# --------------------------------------------------

encrypted_update = aesgcm.encrypt(
    nonce,
    model_update,
    associated_data
)

print("\nAES-256-GCM encryption:")
print("Encryption succeeded: True")
print("Nonce size:", len(nonce), "bytes")
print("Encrypted update size:", len(encrypted_update), "bytes")

# --------------------------------------------------
# 7. AES-256-GCM decryption
# --------------------------------------------------

decrypted_update = aesgcm.decrypt(
    nonce,
    encrypted_update,
    associated_data
)

print("\nAES-256-GCM decryption:")
print("Decryption succeeded: True")
print(
    "Original model update recovered:",
    decrypted_update == model_update
)

# --------------------------------------------------
# 8. Final validation
# --------------------------------------------------

ml_kem_valid = (
    shared_secret_client == shared_secret_server
)

aes_valid = (
    decrypted_update == model_update
)

integration_passed = ml_kem_valid and aes_valid

print("\n=== Final Integration Validation ===")
print("ML-KEM shared secret validation:", ml_kem_valid)
print("AES model update validation:", aes_valid)
print("ML-KEM + AES integration PASSED:", integration_passed)