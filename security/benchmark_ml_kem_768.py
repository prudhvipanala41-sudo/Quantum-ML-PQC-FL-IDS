import json
import statistics
import time
from pathlib import Path

from pqcrypto.kem import ml_kem_768


ITERATIONS = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "ml_kem"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

keygen_times = []
encaps_times = []
decaps_times = []

successful_runs = 0

for i in range(ITERATIONS):
    # Key generation
    start = time.perf_counter()
    public_key, secret_key = ml_kem_768.keygen()
    keygen_times.append(time.perf_counter() - start)

    # Encapsulation
    start = time.perf_counter()
    ciphertext, sender_secret = ml_kem_768.encaps(public_key)
    encaps_times.append(time.perf_counter() - start)

    # Decapsulation
    start = time.perf_counter()
    receiver_secret = ml_kem_768.decaps(secret_key, ciphertext)
    decaps_times.append(time.perf_counter() - start)

    # Validation
    if sender_secret != receiver_secret:
        raise RuntimeError(
            f"Shared-secret verification failed on iteration {i + 1}"
        )

    successful_runs += 1


def statistics_for(values):
    return {
        "min_ms": min(values) * 1000,
        "max_ms": max(values) * 1000,
        "mean_ms": statistics.mean(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
    }


artifact = {
    "algorithm": ml_kem_768.ALGORITHM,
    "iterations": ITERATIONS,
    "successful_runs": successful_runs,
    "validation": {
        "all_shared_secrets_match": successful_runs == ITERATIONS
    },
    "sizes_bytes": {
        "public_key": ml_kem_768.PUBLIC_KEY_SIZE,
        "secret_key": ml_kem_768.SECRET_KEY_SIZE,
        "ciphertext": ml_kem_768.CIPHERTEXT_SIZE,
        "shared_secret": ml_kem_768.SHARED_SECRET_SIZE,
    },
    "timing": {
        "key_generation": statistics_for(keygen_times),
        "encapsulation": statistics_for(encaps_times),
        "decapsulation": statistics_for(decaps_times),
    },
}


output_file = RESULTS_DIR / "ml_kem_768_benchmark.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(artifact, f, indent=4)

print("ML-KEM-768 benchmark completed")
print("-" * 45)
print("Algorithm:", artifact["algorithm"])
print("Iterations:", ITERATIONS)
print("Successful runs:", successful_runs)
print("All shared secrets match:",
      artifact["validation"]["all_shared_secrets_match"])

print("\nSizes:")
for name, size in artifact["sizes_bytes"].items():
    print(f"{name}: {size} bytes")

print("\nTiming:")
for operation, values in artifact["timing"].items():
    print(
        f"{operation}: "
        f"mean={values['mean_ms']:.4f} ms, "
        f"median={values['median_ms']:.4f} ms, "
        f"min={values['min_ms']:.4f} ms, "
        f"max={values['max_ms']:.4f} ms"
    )

print("\nArtifact saved to:")
print(output_file)

if not artifact["validation"]["all_shared_secrets_match"]:
    raise RuntimeError("Benchmark validation failed")

print("\nML-KEM-768 benchmark PASSED")