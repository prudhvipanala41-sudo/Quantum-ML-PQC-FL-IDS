import pandas as pd
import os

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

train_path = os.path.join("data", "raw", "KDDTrain+.txt")
test_path = os.path.join("data", "raw", "KDDTest+.txt")

train = pd.read_csv(train_path, header=None, names=columns)
test = pd.read_csv(test_path, header=None, names=columns)

print("\n========== NSL-KDD DATASET INSPECTION ==========\n")

print("TRAINING DATA")
print("Rows:", train.shape[0])
print("Columns:", train.shape[1])

print("\nTESTING DATA")
print("Rows:", test.shape[0])
print("Columns:", test.shape[1])

print("\n========== COLUMN NAMES ==========\n")

for i, column in enumerate(columns):
    print(i, "->", column)

print("\n========== FIRST 5 TRAINING RECORDS ==========\n")
print(train.head())

print("\n========== DATA TYPES ==========\n")
print(train.dtypes)

print("\n========== MISSING VALUES ==========\n")
print(train.isnull().sum())

print("\n========== TRAINING LABEL COUNTS ==========\n")
print(train["label"].value_counts())

print("\n========== TESTING LABEL COUNTS ==========\n")
print(test["label"].value_counts())

print("\n========== CATEGORICAL FEATURES ==========\n")

categorical_columns = train.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    print(column)

print("\n========== NUMERICAL FEATURES ==========\n")

numerical_columns = train.select_dtypes(exclude=["object"]).columns

for column in numerical_columns:
    print(column)

print("\n========== BASIC STATISTICS ==========\n")
print(train.describe())

print("\n========== INSPECTION COMPLETE ==========\n")