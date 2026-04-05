import torch
import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data files
NORMAL_DATA_PATH = os.path.join(DATA_DIR, "normal.csv")
ATTACK_DATA_PATH = os.path.join(DATA_DIR, "attack.csv")
MERGED_DATA_PATH = os.path.join(DATA_DIR, "merged.csv")

# Models and artifacts
MODEL_PATH = os.path.join(MODELS_DIR, "lstm_autoencoder.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Hyperparameters
SEQUENCE_LENGTH = 50
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-3
N_FEATURES = 51

# Device selection: use cuda:0 if available, otherwise cpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Physical Feature Mapping for SWaT Dataset
FEATURE_MAPPING = {
    "FIT": "Flow Sensor",
    "LIT": "Level Sensor",
    "AIT": "Analyzer Sensor (Water Property)",
    "DPIT": "Differential Pressure Sensor",
    "PIT": "Pressure Sensor",
    "MV": "Motorized Valve",
    "P": "Pump",
    "UV": "Ultraviolet Dechlorinator"
}

def get_feature_group(col_name):
    # Match the prefix
    for prefix, group in FEATURE_MAPPING.items():
        if col_name.startswith(prefix):
            return group
    return "Unknown Component"

def get_threat_type(group):
    if "Sensor" in group:
        return "Sensor Spoofing or Data Injection Attack"
    elif "Valve" in group or "Pump" in group or "Ultraviolet" in group:
        return "Control Logic Attack or Malicious Actuator Command"
    return "Unknown Threat"