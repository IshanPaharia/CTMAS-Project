import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import SCALER_PATH

def load_data(path, is_training=False):
    """
    Load data, clean column names, drop non-feature columns,
    and returns a clean numeric dataframe along with feature names and labels.
    """
    df = pd.read_csv(path, encoding='latin1')
    df.columns = df.columns.str.strip()

    # Drop timestamp if exists
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    labels = None
    if "Normal/Attack" in df.columns:
        # Create binary labels: 1 for Attack, 0 for Normal
        labels = (df["Normal/Attack"] != "Normal").astype(int).values
        df = df.drop(columns=["Normal/Attack"])
    elif "label" in df.columns:
        labels = df["label"].values
        df = df.drop(columns=["label"])

    # Retain feature names
    feature_names = df.columns.tolist()

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)

    # For fast training demo, slice if is_training
    if is_training:
        df = df.iloc[:50000]

    return df, labels, feature_names

def fit_scaler(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    joblib.dump(scaler, SCALER_PATH)
    return scaled, scaler

def transform_scaler(df):
    scaler = joblib.load(SCALER_PATH)
    return scaler.transform(df)

def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)