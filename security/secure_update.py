from pqcrypto.kem import ml_kem_768
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os


def establish_session_key():
    public_key, secret_key = ml_kem_768.keygen()

    kem_ciphertext, client_shared_secret = ml_kem_768.encaps(
        public_key
    )

    server_shared_secret = ml_kem_768.decaps(
        secret_key,
        kem_ciphertext
    )

    if client_shared_secret != server_shared_secret:
        raise ValueError("ML-KEM shared secret mismatch")

    return client_shared_secret, kem_ciphertext


def encrypt_model_update(model_update, session_key, associated_data=b"FL-MODEL-UPDATE"):
    if len(session_key) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")

    aesgcm = AESGCM(session_key)

    nonce = os.urandom(12)

    ciphertext = aesgcm.encrypt(
        nonce,
        model_update,
        associated_data
    )

    return {
        "ciphertext": ciphertext,
        "nonce": nonce,
        "associated_data": associated_data
    }


def decrypt_model_update(encrypted_update, session_key):
    if len(session_key) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")

    aesgcm = AESGCM(session_key)

    plaintext = aesgcm.decrypt(
        encrypted_update["nonce"],
        encrypted_update["ciphertext"],
        encrypted_update["associated_data"]
    )

    return plaintext


def verify_secure_update(model_update):
    session_key, kem_ciphertext = establish_session_key()

    encrypted_update = encrypt_model_update(
        model_update,
        session_key
    )

    decrypted_update = decrypt_model_update(
        encrypted_update,
        session_key
    )

    return {
        "ml_kem_ciphertext_size": len(kem_ciphertext),
        "session_key_size": len(session_key),
        "nonce_size": len(encrypted_update["nonce"]),
        "ciphertext_size": len(encrypted_update["ciphertext"]),
        "encryption_success": True,
        "decryption_success": True,
        "data_integrity": decrypted_update == model_update
    }


if __name__ == "__main__":
    test_update = b"client model parameters"

    result = verify_secure_update(test_update)

    print("=== Secure Model Update Test ===")
    print("ML-KEM ciphertext size:", result["ml_kem_ciphertext_size"], "bytes")
    print("Session key size:", result["session_key_size"], "bytes")
    print("AES-GCM nonce size:", result["nonce_size"], "bytes")
    print("Encrypted update size:", result["ciphertext_size"], "bytes")
    print("Encryption success:", result["encryption_success"])
    print("Decryption success:", result["decryption_success"])
    print("Data integrity:", result["data_integrity"])

    passed = all([
        result["encryption_success"],
        result["decryption_success"],
        result["data_integrity"]
    ])

    print("Secure update validation:", passed)