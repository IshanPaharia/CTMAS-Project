# Explainable AI (XAI) Based Threat Modeling

A robust anomaly detection and cyber-threat modeling system designed for Industrial Control Systems (ICS). This project implements an **LSTM Autoencoder** to identify behavioral deviations in sensor and actuator traffic, with integrated explainability to map those anomalies to specific physical components (e.g., Pumps, Valves).

## 🚀 Key Features

*   **Intelligent Detection**: Uses an LSTM Autoencoder trained on normal time-series data to detect complex, non-linear anomalies.
*   **Explainable AI (XAI)**: Integrated Gradients are used to pinpoint WHICH specific sensor or actuator is causing the anomaly.
*   **Semantic Reasoner**: Maps technical sensor IDs (e.g., `P101`, `MV201`) to real-world roles to calculate if the threat is a "Sensor Spoofing" or "Actuator Hijacking" attempt.
*   **Adversarial Defense**: A separate demonstration (`main_ADVTraining.py`) utilizing **Adversarial Training (PGD)** to defend against sophisticated ML evasion attacks.

## 📁 Repository Structure

*   `main.py`: The standard demonstration (Vulnerable to Evasion).
*   `main_ADVTraining.py`: The robust demonstration (Defends against Evasion).
*   `src/train.py`: Baseline training logic.
*   `src/train_ADVTraining.py`: Advanced robust training using adversarial noise injection.
*   `src/detect.py`: Detectors and Feature Squeezing defenses.
*   `src/explainer.py`: The Integrated Gradients engine for XAI.
*   `src/config.py`: Physical-to-Semantic device mapping.

## 🛠️ Getting Started

### 1. Requirements
Ensure you have Python 3.8+ and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execution
*   **Run Baseline Demo**:
    `python main.py`
*   **Run Robust Demo**:
    `python main_ADVTraining.py`

## 🛡️ Adversarial Knowledge
This project explores the "White-Box" attack model where an attacker (using **PGD**) attempts to minimize the reconstruction error of malicious traffic to bypass the IDS. The `ADVTraining` version counters this using a combined Denoising/Robust loss function.

## 📊 Mapping Documentation
Device tags follow the CTMAS dataset standards:
- `P***`: Pumps (Actuators)
- `MV***`: Motorized Valves (Actuators)
- `LIT*** / PIT***`: Level/Pressure Indicators (Sensors)
- `AIT***`: Flow/Analytical Indicators (Sensors)
