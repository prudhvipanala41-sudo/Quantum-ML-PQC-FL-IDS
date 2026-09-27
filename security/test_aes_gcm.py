from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os


print("=== AES-256-GCM Standalone Test ===")

# Generate a 256-bit AES key
key = AESGCM.generate_key(bit_length=256)

# Create AES-GCM object
aesgcm = AESGCM(key)

# Random 12-byte nonce
nonce = os.urandom(12)

# Simulated model update
original_data = b"Federated learning model update: client_1_weights"

# Optional authenticated metadata
associated_data = b"FL-Client-1"

print("\nOriginal data:")
print(original_data)

# Encryption
ciphertext = aesgcm.encrypt(
    nonce,
    original_data,
    associated_data
)

print("\nEncryption:")
print("Encryption succeeded: True")
print("Nonce size:", len(nonce), "bytes")
print("Ciphertext size:", len(ciphertext), "bytes")

# Decryption
decrypted_data = aesgcm.decrypt(
    nonce,
    ciphertext,
    associated_data
)

print("\nDecryption:")
print("Decryption succeeded: True")
print("Original data recovered:", decrypted_data == original_data)

# Tampering test
tampered_ciphertext = bytearray(ciphertext)
tampered_ciphertext[0] ^= 1

print("\nTampering test:")

try:
    aesgcm.decrypt(
        nonce,
        bytes(tampered_ciphertext),
        associated_data
    )
    print("Authentication failure detected: False")
except Exception:
    print("Authentication failure detected: True")

# Final validation
passed = (
    decrypted_data == original_data
    and len(nonce) == 12
)

print("\n=== Final Validation ===")
print("AES-256-GCM standalone test PASSED:", passed)