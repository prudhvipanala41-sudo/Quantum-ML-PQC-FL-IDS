import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "data/raw/KDDTrain+.txt"
TEST_PATH = "data/raw/KDDTest+.txt"

OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# COLUMN NAMES
# ============================================================

columns = [
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
# LOAD DATA
# ============================================================

print("Loading training data...")
train_df = pd.read_csv(
    TRAIN_PATH,
    header=None,
    names=columns
)

print("Loading testing data...")
test_df = pd.read_csv(
    TEST_PATH,
    header=None,
    names=columns
)

print("Training shape:", train_df.shape)
print("Testing shape :", test_df.shape)


# ============================================================
# REMOVE DATASET METADATA
# ============================================================

print("\nRemoving difficulty column...")

train_df = train_df.drop(columns=["difficulty"])
test_df = test_df.drop(columns=["difficulty"])


# ============================================================
# CONVERT LABEL INTO BINARY IDS TARGET
# ============================================================

print("\nConverting labels...")

train_df["label"] = train_df["label"].apply(
    lambda x: 0 if x == "normal" else 1
)

test_df["label"] = test_df["label"].apply(
    lambda x: 0 if x == "normal" else 1
)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]


print("\nFeature columns:", X_train.shape[1])
print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "protocol_type",
    "service",
    "flag"
]

numerical_features = [
    column
    for column in X_train.columns
    if column not in categorical_features
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumber of numerical features:", len(numerical_features))


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)


# ============================================================
# FIT ONLY ON TRAINING DATA
# ============================================================

print("\nFitting preprocessing pipeline on training data...")

X_train_processed = preprocessor.fit_transform(X_train)


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

print("Transforming testing data...")

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# CONVERT LABELS TO NUMPY
# ============================================================

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

print("\nSaving processed data...")

np.save(
    os.path.join(OUTPUT_DIR, "X_train.npy"),
    X_train_processed
)

np.save(
    os.path.join(OUTPUT_DIR, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(OUTPUT_DIR, "X_test.npy"),
    X_test_processed
)

np.save(
    os.path.join(OUTPUT_DIR, "y_test.npy"),
    y_test
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    os.path.join(OUTPUT_DIR, "preprocessor.joblib")
)


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_info = {
    "original_feature_count": len(X_train.columns),
    "processed_feature_count": X_train_processed.shape[1],
    "categorical_features": categorical_features,
    "numerical_feature_count": len(numerical_features),
    "classification": {
        "normal": 0,
        "attack": 1
    }
}

with open(
    os.path.join(OUTPUT_DIR, "feature_info.json"),
    "w"
) as file:
    json.dump(feature_info, file, indent=4)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n====================================")
print("PREPROCESSING COMPLETED")
print("====================================")

print("Original features :", X_train.shape[1])
print("Processed features:", X_train_processed.shape[1])

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)

print("\nTraining labels:")
print(np.unique(y_train, return_counts=True))

print("\nTesting labels:")
print(np.unique(y_test, return_counts=True))

print("\nFiles saved inside:")
print(OUTPUT_DIR)