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

    def compute_reconstruction_error(self, sequences, batch_size=256, per_feature=False, defense_mode=False, smoothing_samples=0):
        """
        Computes the MSE reconstruction error.
        
        Rigidity Features:
        - defense_mode: Applies Feature Squeezing (rounding) to break precision-based attacks.
        - smoothing_samples: If > 0, applies Randomized Smoothing (averaging over noise). 
          This is the most 'rigid' empirical defense available.
        """
        errors = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_raw = sequences[i:i + batch_size]
                batch = torch.tensor(batch_raw, dtype=torch.float32).to(self.device)
                
                if defense_mode:
                    # Feature Squeezing: Reduces the degrees of freedom for the attacker
                    batch = torch.round(batch * 50) / 50.0

                if smoothing_samples > 0:
                    # --- RANDOMIZED SMOOTHING (RIGID DEFENSE) ---
                    # We create multiple noisy versions and average their reconstruction error
                    # This smooths the manifold and makes it nearly impossible to find a "gap"
                    cumulative_loss = torch.zeros(batch.shape[0], device=self.device) if not per_feature else \
                                      torch.zeros(batch.shape[0], batch.shape[2], device=self.device)
                    
                    sigma = 0.02 # Noise level
                    for _ in range(smoothing_samples):
                        noise = torch.randn_like(batch) * sigma
                        noisy_batch = torch.clamp(batch + noise, 0.0, 1.0)
                        outputs = self.model(noisy_batch)
                        loss = (outputs - noisy_batch) ** 2
                        
                        if per_feature:
                            cumulative_loss += loss.mean(dim=1)
                        else:
                            cumulative_loss += loss.mean(dim=(1, 2))
                    
                    batch_errors = cumulative_loss / smoothing_samples
                else:
                    # Standard computation
                    outputs = self.model(batch)
                    loss = (outputs - batch) ** 2
                    
                    if per_feature:
                        batch_errors = loss.mean(dim=1)
                    else:
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