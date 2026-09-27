from pqcrypto.kem import ml_kem_768


def main():
    print("ML-KEM-768 standalone test")
    print("-" * 40)

    print("Algorithm:", ml_kem_768.ALGORITHM)
    print("Public key size:", ml_kem_768.PUBLIC_KEY_SIZE)
    print("Secret key size:", ml_kem_768.SECRET_KEY_SIZE)
    print("Ciphertext size:", ml_kem_768.CIPHERTEXT_SIZE)
    print("Shared secret size:", ml_kem_768.SHARED_SECRET_SIZE)

    print("\n[1] Generating key pair...")
    public_key, secret_key = ml_kem_768.keygen()

    print("Public key generated:", len(public_key), "bytes")
    print("Secret key generated:", len(secret_key), "bytes")

    print("\n[2] Encapsulating shared secret...")
    ciphertext, shared_secret_sender = ml_kem_768.encaps(public_key)

    print("Ciphertext generated:", len(ciphertext), "bytes")
    print(
        "Sender shared secret generated:",
        len(shared_secret_sender),
        "bytes",
    )

    print("\n[3] Decapsulating shared secret...")
    shared_secret_receiver = ml_kem_768.decaps(secret_key, ciphertext)

    print(
        "Receiver shared secret generated:",
        len(shared_secret_receiver),
        "bytes",
    )

    print("\n[4] Verifying shared-secret equality...")
    secrets_match = shared_secret_sender == shared_secret_receiver

    print("Shared secrets match:", secrets_match)

    if not secrets_match:
        raise RuntimeError("ML-KEM-768 shared-secret verification failed")

    print("\nML-KEM-768 standalone test PASSED")


if __name__ == "__main__":
    main()