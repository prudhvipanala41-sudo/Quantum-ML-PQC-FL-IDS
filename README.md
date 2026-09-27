# Quantum-ML-PQC-FL-IDS

A cutting-edge research project and prototype that integrates Quantum Machine Learning (QML), Post-Quantum Cryptography (PQC), and Federated Learning (FL) to build a secure, robust, and future-proof Intrusion Detection System (IDS).

## 🚀 Overview

This project explores the intersection of quantum computing, advanced cryptography, and distributed AI to tackle modern cybersecurity challenges. It aims to detect network intrusions while protecting the machine learning training process from both classical attacks (e.g., model poisoning) and future quantum computing threats.

### Key Features
- **Quantum Machine Learning (QML)**: Utilizes Quantum Kernel Support Vector Machines (QKSVM) via [PennyLane](https://pennylane.ai/) to enhance the detection capabilities of the IDS.
- **Post-Quantum Cryptography (PQC)**: Secures communication and model updates in the federated network using quantum-resistant algorithms like ML-KEM-768 and AES-GCM.
- **Federated Learning (FL)**: Enables decentralized model training (e.g., FedAvg) across multiple clients, ensuring data privacy and localized learning.
- **Robust Aggregation**: Defends against adversarial attacks like model poisoning to maintain the integrity of the global model.
- **Web Dashboard**: A Flask-based web application to visualize benchmarking metrics, evaluation summaries, and final model comparisons.

## 📁 Project Structure

- `app/`: Flask web application and UI for visualizing results.
- `attacks/`: Scripts and utilities for simulating network or adversarial ML attacks.
- `data/`: Datasets for training and evaluating the IDS (processed and reduced).
- `evaluation/`: Scripts for evaluating model performance and calculating metrics.
- `experiments/`: Configurations and scripts for running experimental scenarios.
- `federated/`: Federated learning implementation, including FedAvg and local training mechanisms.
- `preprocessing/`: Data preprocessing, normalization, and feature engineering pipelines.
- `qml/`: Quantum Machine Learning models, specifically QKSVM architectures.
- `results/`: Generated outputs, benchmarks, JSON summaries, and graphs.
- `security/`: Post-Quantum Cryptographic implementations, secure FL transport, and robust aggregation logic.
- `tests/`: Unit and integration tests for various modules.

## 🛠️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd Quantum-ML-PQC-FL-IDS
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

### Running the Web Dashboard
You can view the results of the experiments through the web interface:

```bash
python app/app.py
```
Then, open your browser and navigate to `http://127.0.0.1:5000`.

### Running Experiments
*(Note: Refer to specific scripts in the project directories to run pipelines)*
- **QML Evaluation**: `python qml/qksvm_ids.py`
- **Federated Training**: `python federated/fedavg.py`
- **Security Benchmarking**: Scripts under `security/` (e.g., `python security/benchmark_ml_kem_768.py`)

## 🛡️ Security

This project implements ML-KEM-768 for key encapsulation and AES-GCM for authenticated encryption. This ensures that all federated model updates remain confidential and tamper-proof against both classical and quantum adversaries.

## 📄 License
[MIT License] - *Please update according to your requirements.*
