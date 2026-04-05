import torch
import numpy as np
import pandas as pd

from config import *
from preprocessing import load_data, transform_scaler, create_sequences
from model import LSTMAutoencoder

class AnomalyDetector:
    def __init__(self, use_global_threshold=True, model_path=None):
        self.device = DEVICE
        self.model = LSTMAutoencoder(N_FEATURES).to(self.device)
        if model_path is None:
            model_path = MODEL_PATH
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.threshold = None
        self.use_global_threshold = use_global_threshold

    def compute_reconstruction_error(self, sequences, batch_size=256, per_feature=False, defense_mode=False):
        """
        Computes the MSE reconstruction error.
        If per_feature is True, returns errors shaped (N, num_features) representing the sum of errors across sequence length per feature.
        Otherwise returns global anomalies shaped (N,).
        If defense_mode is True, applies robust Feature Squeezing to break FGSM gradients.
        """
        errors = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                
                if defense_mode:
                    batch = torch.round(batch * 100) / 100.0

                outputs = self.model(batch)
                loss = (outputs - batch) ** 2
                
                if per_feature:
                    # shape: (batch_size, seq_len, features) -> mean across seq_len
                    batch_errors = loss.mean(dim=1)
                else:
                    # global error over seq_len and features
                    batch_errors = loss.mean(dim=(1, 2))
                
                errors.extend(batch_errors.cpu().numpy())

        return np.array(errors)

    def fit_threshold(self, normal_df):
        """
        Calculates the 95th percentile error of normal data for the threshold.
        """
        print("[*] Computing anomaly threshold on normal data...")
        normal_scaled = transform_scaler(normal_df)
        normal_seq = create_sequences(normal_scaled, SEQUENCE_LENGTH)
        
        normal_errors = self.compute_reconstruction_error(normal_seq, per_feature=False)
        self.threshold = np.percentile(normal_errors, 95)
        print(f"    -> Threshold set to: {self.threshold:.6f}")
        return self.threshold

    def detect(self, df):
        """
        Detect anomalies against the fitted threshold.
        Returns the boolean predictions and the global errors.
        """
        if self.threshold is None:
            raise ValueError("Threshold is not fitted. Call fit_threshold first.")

        # In case the df isn't scaled yet:
        scaled = transform_scaler(df)
        seq = create_sequences(scaled, SEQUENCE_LENGTH)
        
        errors = self.compute_reconstruction_error(seq, per_feature=False)
        predictions = (errors > self.threshold).astype(int)
        
        return predictions, errors

    def get_most_anomalous_features(self, sequence, feature_names, top_k=3):
        """
        Computes the per-feature reconstruction error for a single sequence and returns the top_k feature names.
        sequence shape should be (1, seq_len, num_features)
        """
        with torch.no_grad():
            seq_tensor = torch.tensor(sequence, dtype=torch.float32).to(self.device)
            # Add batch dimension if necessary
            if len(seq_tensor.shape) == 2:
                seq_tensor = seq_tensor.unsqueeze(0)
                
            recon = self.model(seq_tensor)
            error = ((seq_tensor - recon) ** 2).mean(dim=1).squeeze(0) # shape: (num_features,)
            
            error_np = error.cpu().numpy()
            
            # Find indices of largest errors
            top_indices = error_np.argsort()[-top_k:][::-1]
            anomalous_features = [(feature_names[i], error_np[i]) for i in top_indices]
            
            return anomalous_features