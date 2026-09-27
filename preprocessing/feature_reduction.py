import os
import json
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump


# ============================================================
# SETTINGS
# ============================================================

X_TRAIN_PATH = "data/processed/X_train.npy"
Y_TRAIN_PATH = "data/processed/y_train.npy"

X_TEST_PATH = "data/processed/X_test.npy"
Y_TEST_PATH = "data/processed/y_test.npy"

PARTITION_IID_DIR = "data/partitions/iid"
PARTITION_NON_IID_DIR = "data/partitions/non_iid"

OUTPUT_DIR = "data/processed/reduced"

K_FEATURES = 8


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TRAINING AND TEST DATA
# ============================================================

print("Loading processed training data...")

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)

print("Training X shape:", X_train.shape)
print("Training y shape:", y_train.shape)


print("\nLoading processed testing data...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("Testing X shape:", X_test.shape)
print("Testing y shape:", y_test.shape)


# ============================================================
# FEATURE SELECTION
# ============================================================

print("\n====================================")
print("FEATURE REDUCTION")
print("====================================")

print("Original number of features:", X_train.shape[1])
print("Selecting:", K_FEATURES, "features")


selector = SelectKBest(
    score_func=f_classif,
    k=K_FEATURES
)


# IMPORTANT:
# Fit selector ONLY on training data.

X_train_reduced = selector.fit_transform(
    X_train,
    y_train
)


# Apply the SAME selector to test data.

X_test_reduced = selector.transform(
    X_test
)


print("\nReduced training shape:")
print(X_train_reduced.shape)

print("Reduced testing shape:")
print(X_test_reduced.shape)


# ============================================================
# GET FEATURE SCORES
# ============================================================

scores = selector.scores_
selected_indices = selector.get_support(indices=True)


print("\n====================================")
print("SELECTED FEATURES")
print("====================================")

print("Selected feature indices:")

print(selected_indices)


print("\nFeature scores:")

for index in selected_indices:

    print(
        "Feature",
        index,
        "Score =",
        scores[index]
    )


# ============================================================
# SAVE REDUCED TRAINING / TESTING DATA
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train_reduced.npy"
    ),
    X_train_reduced
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train_reduced.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test_reduced.npy"
    ),
    X_test_reduced
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test_reduced.npy"
    ),
    y_test
)


# ============================================================
# SAVE FEATURE SELECTOR
# ============================================================

dump(
    selector,
    os.path.join(
        OUTPUT_DIR,
        "feature_selector.joblib"
    )
)


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_information = {

    "original_features": int(
        X_train.shape[1]
    ),

    "selected_features": int(
        K_FEATURES
    ),

    "selected_feature_indices": [
        int(x)
        for x in selected_indices
    ],

    "feature_scores": [
        float(scores[x])
        for x in selected_indices
    ]
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "feature_info.json"
    ),
    "w"
) as file:

    json.dump(
        feature_information,
        file,
        indent=4
    )


# ============================================================
# REDUCE IID CLIENT DATA
# ============================================================

print("\n====================================")
print("REDUCING IID CLIENT DATA")
print("====================================")


for client_number in range(1, 6):

    input_path = os.path.join(
        PARTITION_IID_DIR,
        f"client_{client_number}_X.npy"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "iid",
        f"client_{client_number}_X.npy"
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    client_X = np.load(input_path)

    client_X_reduced = selector.transform(
        client_X
    )

    np.save(
        output_path,
        client_X_reduced
    )

    print(
        f"Client {client_number}:",
        client_X.shape,
        "->",
        client_X_reduced.shape
    )


# ============================================================
# REDUCE NON-IID CLIENT DATA
# ============================================================

print("\n====================================")
print("REDUCING NON-IID CLIENT DATA")
print("====================================")


for client_number in range(1, 6):

    input_path = os.path.join(
        PARTITION_NON_IID_DIR,
        f"client_{client_number}_X.npy"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "non_iid",
        f"client_{client_number}_X.npy"
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    client_X = np.load(input_path)

    client_X_reduced = selector.transform(
        client_X
    )

    np.save(
        output_path,
        client_X_reduced
    )

    print(
        f"Client {client_number}:",
        client_X.shape,
        "->",
        client_X_reduced.shape
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")
print("FEATURE REDUCTION COMPLETED")
print("====================================")

print(
    "Original features :",
    X_train.shape[1]
)

print(
    "Reduced features  :",
    X_train_reduced.shape[1]
)

print(
    "Training samples  :",
    X_train_reduced.shape[0]
)

print(
    "Testing samples   :",
    X_test_reduced.shape[0]
)

print(
    "\nReduced data saved in:"
)

print(
    OUTPUT_DIR
)