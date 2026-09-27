# Quantum ML-Based Intrusion Detection with PQC-Secured Federated Model Updates

A security-focused research project that combines **Intrusion Detection Systems (IDS)**, **Quantum Machine Learning (QML)**, **Federated Learning (FL)**, **Post-Quantum Cryptography (PQC)**, and **adversarial robustness**.

The project investigates how distributed intrusion detection models can be trained collaboratively while protecting model updates using **ML-KEM-768** for post-quantum key establishment and **AES-256-GCM** for authenticated encryption.

---

## 📌 Project Overview

Traditional centralized intrusion detection systems require collecting data at a central location. This can create privacy, communication, and security concerns.

This project implements a complete experimental pipeline:

```text
NSL-KDD Dataset
       ↓
Dataset Inspection
       ↓
Preprocessing
       ↓
Feature Reduction
       ↓
Classical IDS + QML IDS
       ↓
Federated Learning
       ↓
ML-KEM-768 Key Establishment
       ↓
AES-256-GCM Model Update Encryption
       ↓
Secure Federated Learning
       ↓
Model Poisoning Attack
       ↓
Robust Median Aggregation
       ↓
Final Evaluation
       ↓
Graphs & Security Benchmarking
       ↓
Flask-Based Project Dashboard
