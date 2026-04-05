import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from config import DEVICE, get_feature_group, get_threat_type

class XAIEngine:
    def __init__(self, model):
        """
        Initializes the Explainable AI Engine using Captum.
        model: PyTorch LSTMAutoencoder module.
        """
        self.model = model.to(DEVICE)
        self.model.eval()
        self.device = DEVICE

    def _model_output_wrapper(self, x):
        """
        Wrapper to compute sequence mean squared error, maintaining batch dimension.
        Output shape should be (batch_size,) indicating importance target.
        """
        # Compute reconstruction error per sample in the batch
        recon = self.model(x)
        error = torch.mean((x - recon) ** 2, dim=(1, 2))
        return error

    def get_integrated_gradients(self, sample):
        """
        Calculates Integrated Gradients for a single [1, seq_len, num_features] sample tensor.
        """
        self.model.train()  # As per original notebook: Captum required .train() for some internal gradients on LSTM
        
        # Ensure gradient requirements
        sample = sample.clone().detach().to(self.device).requires_grad_(True)
        
        ig = IntegratedGradients(self._model_output_wrapper)
        baseline = torch.zeros_like(sample)
        
        attr, _ = ig.attribute(sample, baselines=baseline, return_convergence_delta=True)
        
        # Returning back to eval
        self.model.eval()
        
        return attr.detach().cpu().numpy()

    def plot_ig_heatmap(self, ig_attr, feature_names):
        """
        Plots a heatmap of the Integrated Gradients attribution across the sequence.
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(ig_attr[0].T, aspect='auto', cmap='hot')
        plt.colorbar(label='Feature Attribution (Gradient Magnitude)')
        plt.title("XAI Integrated Gradients Feature Importance Over Time")
        plt.xlabel("Sequence Timestep")
        plt.ylabel("Feature Index")
        plt.savefig("xai_heatmap.png")
        print("\n    -> Saved XAI Heatmap visualization to 'xai_heatmap.png'.")

    def semantic_threat_reasoning(self, anomalous_features):
        """
        Maps top anomalous features to qualitative threat definitions based on physical components.
        """
        print("\n--- XAI SEMANTIC THREAT REASONING ---")
        
        # Group anomalous features
        threat_groups = []
        for feature, error in anomalous_features:
            group = get_feature_group(feature)
            type_of_threat = get_threat_type(group)
            threat_groups.append(type_of_threat)
            print(f"[*] Found massive deviation in [{feature}] -> Mapped to: {group}")
        
        # Determine the primary threat footprint
        if "Control Logic Attack or Malicious Actuator Command" in threat_groups:
            print("\n[!] CRITICAL ANALYSIS: The XAI strongly footprinted an ACTUATOR anomaly.")
            print("    -> Conclusion: The adversary is likely attempting a Physical Process Intervention (e.g. Forcing Pumps/Valves On/Off).")
        elif "Sensor Spoofing or Data Injection Attack" in threat_groups:
            print("\n[!] CRITICAL ANALYSIS: The XAI strongly footprinted a SENSOR anomaly.")
            print("    -> Conclusion: The adversary is likely masking data or spoofing a sensor readout (e.g. Replay Attack).")
        else:
            print("\n[!] CRITICAL ANALYSIS: The XAI found anomalous deviations in unmapped/generic features.")
            print("    -> Conclusion: Multi-dimensional complex threat vector.")
        print("---------------------------------------")
