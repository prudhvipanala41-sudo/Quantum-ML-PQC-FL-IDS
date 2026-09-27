import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

NUM_CLIENTS = 5
RANDOM_SEED = 42

RAW_TRAIN_PATH = "data/raw/KDDTrain+.txt"
X_PATH = "data/processed/X_train.npy"
Y_PATH = "data/processed/y_train.npy"

IID_DIR = "data/partitions/iid"
NON_IID_DIR = "data/partitions/non_iid"


# ============================================================
# NSL-KDD COLUMN NAMES
# ============================================================

COLUMN_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]


# ============================================================
# ATTACK CATEGORIES
# ============================================================

ATTACK_CATEGORIES = {
    "dos": [
        "neptune",
        "back",
        "land",
        "pod",
        "smurf",
        "teardrop"
    ],

    "probe": [
        "ipsweep",
        "nmap",
        "portsweep",
        "satan"
    ],

    "r2l": [
        "ftp_write",
        "guess_passwd",
        "imap",
        "multihop",
        "phf",
        "warezclient",
        "warezmaster"
    ],

    "u2r": [
        "buffer_overflow",
        "loadmodule",
        "perl",
        "rootkit"
    ]
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(IID_DIR, exist_ok=True)
os.makedirs(NON_IID_DIR, exist_ok=True)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading raw training data...")

raw_data = pd.read_csv(
    RAW_TRAIN_PATH,
    header=None,
    names=COLUMN_NAMES
)

print("Raw training shape:", raw_data.shape)


print("\nLoading processed features...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Processed X shape:", X.shape)
print("Processed y shape:", y.shape)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(raw_data) != len(X) or len(X) != len(y):
    raise ValueError(
        "Raw data, X and y do not contain the same number of samples."
    )


# ============================================================
# CREATE ATTACK CATEGORY
# ============================================================

def get_attack_category(label):

    if label == "normal":
        return "normal"

    for category, labels in ATTACK_CATEGORIES.items():

        if label in labels:
            return category

    return "other"


raw_data["attack_category"] = raw_data["label"].apply(
    get_attack_category
)


print("\nOverall attack-category distribution:")

print(
    raw_data["attack_category"].value_counts()
)


# ============================================================
# GET INDICES FOR EACH CATEGORY
# ============================================================

category_indices = {}

for category in [
    "normal",
    "dos",
    "probe",
    "r2l",
    "u2r",
    "other"
]:

    indices = np.where(
        raw_data["attack_category"].values == category
    )[0]

    indices = indices.copy()

    rng.shuffle(indices)

    category_indices[category] = indices


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_client(
    directory,
    client_number,
    indices
):

    client_X = X[indices]
    client_y = y[indices]

    np.save(
        os.path.join(
            directory,
            f"client_{client_number}_X.npy"
        ),
        client_X
    )

    np.save(
        os.path.join(
            directory,
            f"client_{client_number}_y.npy"
        ),
        client_y
    )


# ============================================================
# IID PARTITION
# ============================================================

print("\n====================================")
print("CREATING IID PARTITIONS")
print("====================================")


all_indices = np.arange(len(X))

rng.shuffle(all_indices)

iid_clients = np.array_split(
    all_indices,
    NUM_CLIENTS
)


for client_number, indices in enumerate(
    iid_clients,
    start=1
):

    save_client(
        IID_DIR,
        client_number,
        indices
    )


# ============================================================
# NON-IID PARTITION
# ============================================================

print("\n====================================")
print("CREATING NON-IID PARTITIONS")
print("====================================")


# ------------------------------------------------------------
# Strategy:
#
# We distribute each category independently.
#
# Normal data:
#     distributed approximately equally
#
# Attack categories:
#     different proportions are given to different clients
#
# This creates heterogeneous/non-IID clients while ensuring
# every sample is used exactly once.
# ------------------------------------------------------------


# Client attack preference matrix
#
# Rows    = clients
# Columns = dos, probe, r2l, u2r, other
#
# Values represent the approximate fraction of each category
# assigned to each client.

distribution = np.array([

    [0.70, 0.05, 0.05, 0.05, 0.10],

    [0.10, 0.55, 0.10, 0.10, 0.10],

    [0.10, 0.10, 0.55, 0.10, 0.10],

    [0.10, 0.10, 0.10, 0.65, 0.10],

    [0.00, 0.20, 0.20, 0.10, 0.60]

])


# ------------------------------------------------------------
# Normalize columns
#
# Each attack category must sum to 100% across clients.
# ------------------------------------------------------------

distribution = distribution / distribution.sum(
    axis=0,
    keepdims=True
)


# ------------------------------------------------------------
# Store client indices
# ------------------------------------------------------------

non_iid_clients = [
    [] for _ in range(NUM_CLIENTS)
]


# ============================================================
# DISTRIBUTE NORMAL DATA
# ============================================================

normal_indices = category_indices["normal"].copy()

normal_splits = np.array_split(
    normal_indices,
    NUM_CLIENTS
)

for client_number in range(NUM_CLIENTS):

    non_iid_clients[client_number].extend(
        normal_splits[client_number].tolist()
    )


# ============================================================
# DISTRIBUTE ATTACK CATEGORIES
# ============================================================

attack_categories = [
    "dos",
    "probe",
    "r2l",
    "u2r",
    "other"
]


for category_column, category in enumerate(
    attack_categories
):

    indices = category_indices[category].copy()

    total_samples = len(indices)

    start = 0

    for client_number in range(NUM_CLIENTS):

        if client_number == NUM_CLIENTS - 1:

            end = total_samples

        else:

            count = int(
                round(
                    total_samples
                    * distribution[
                        client_number,
                        category_column
                    ]
                )
            )

            end = start + count

        selected = indices[start:end]

        non_iid_clients[
            client_number
        ].extend(
            selected.tolist()
        )

        start = end


# ============================================================
# SHUFFLE EACH CLIENT
# ============================================================

for client_number in range(NUM_CLIENTS):

    non_iid_clients[
        client_number
    ] = np.array(
        non_iid_clients[client_number],
        dtype=int
    )

    rng.shuffle(
        non_iid_clients[client_number]
    )


# ============================================================
# INTEGRITY CHECK
# ============================================================

print("\n====================================")
print("CHECKING NON-IID DATA INTEGRITY")
print("====================================")


# Combine all client indices

combined_indices = np.concatenate(
    non_iid_clients
)


# Check total number of samples

print(
    "Original samples :",
    len(X)
)

print(
    "Assigned samples :",
    len(combined_indices)
)


if len(combined_indices) != len(X):

    raise ValueError(
        "ERROR: Not all samples were assigned."
    )


# Check duplicates

unique_indices = np.unique(
    combined_indices
)

print(
    "Unique samples   :",
    len(unique_indices)
)


if len(unique_indices) != len(X):

    raise ValueError(
        "ERROR: Duplicate samples detected."
    )


# Check missing samples

expected_indices = set(
    range(len(X))
)

actual_indices = set(
    combined_indices
)

missing_indices = (
    expected_indices - actual_indices
)


print(
    "Missing samples  :",
    len(missing_indices)
)


if len(missing_indices) != 0:

    raise ValueError(
        "ERROR: Some samples are missing."
    )


print(
    "\nNON-IID INTEGRITY CHECK PASSED"
)


# ============================================================
# SAVE IID CLIENTS
# ============================================================

print("\nSaving IID clients...")


for client_number, indices in enumerate(
    iid_clients,
    start=1
):

    save_client(
        IID_DIR,
        client_number,
        indices
    )

    client_categories = (
        raw_data.iloc[indices][
            "attack_category"
        ]
        .value_counts()
        .to_dict()
    )

    normal_count = np.sum(
        y[indices] == 0
    )

    attack_count = np.sum(
        y[indices] == 1
    )

    print(
        f"\nClient {client_number}"
    )

    print(
        "Total samples :",
        len(indices)
    )

    print(
        "Normal        :",
        normal_count
    )

    print(
        "Attack        :",
        attack_count
    )

    print(
        "Attack categories:"
    )

    print(
        client_categories
    )


# ============================================================
# SAVE NON-IID CLIENTS
# ============================================================

print("\nSaving Non-IID clients...")


for client_number, indices in enumerate(
    non_iid_clients,
    start=1
):

    save_client(
        NON_IID_DIR,
        client_number,
        indices
    )

    client_categories = (
        raw_data.iloc[indices][
            "attack_category"
        ]
        .value_counts()
        .to_dict()
    )

    normal_count = np.sum(
        y[indices] == 0
    )

    attack_count = np.sum(
        y[indices] == 1
    )

    print(
        f"\nClient {client_number}"
    )

    print(
        "Total samples :",
        len(indices)
    )

    print(
        "Normal        :",
        normal_count
    )

    print(
        "Attack        :",
        attack_count
    )

    print(
        "Attack categories:"
    )

    print(
        client_categories
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")
print("PARTITIONING COMPLETED")
print("====================================")

print(
    "Number of clients:",
    NUM_CLIENTS
)

print(
    "IID folder:"
)

print(
    IID_DIR
)

print(
    "\nNon-IID folder:"
)

print(
    NON_IID_DIR
)

print(
    "\nAll training samples were assigned exactly once."
)