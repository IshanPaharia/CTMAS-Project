import sys
import os
import warnings

# Suppress warnings (like sklearn InconsistentVersionWarning)
warnings.filterwarnings("ignore")

# Append src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import NORMAL_DATA_PATH, ATTACK_DATA_PATH, SEQUENCE_LENGTH
from preprocessing import load_data, transform_scaler, create_sequences
from detect import AnomalyDetector
from explainer import XAIEngine
from adversarial import AdversarialAttacker

def run_demonstration():
    print("=" * 65)
    print(" 🛡️  EXPLAINABLE AI-BASED THREAT MODELING DEMONSTRATION  🛡️")
    print("=" * 65 + "\n")

    # 1. Initialize Anomaly Detector & Load Pretrained Model
    print("🔹 STEP 1 | INITIALIZING SYSTEM")
    print("-" * 65)
    print("  ⏳ Loading LSTM Autoencoder...")
    detector = AnomalyDetector()
    print("  ✅ System Initialized.\n")

    # 2. Fit Threshold on Normal Data
    print("🔹 STEP 2 | NORMAL TRAFFIC PROFILING")
    print("-" * 65)
    normal_df, _, feature_names = load_data(NORMAL_DATA_PATH, is_training=True)
    threshold = detector.fit_threshold(normal_df)
    print("  ✅ Profiling Complete.\n")
    
    # 3. Detect Anomalies in Attack Data
    print("🔹 STEP 3 | REAL-TIME THREAT DETECTION")
    print("-" * 65)
    print(f"  📡 Simulating Cyber-Physical Feed [{ATTACK_DATA_PATH}]")
    attack_df, _, _ = load_data(ATTACK_DATA_PATH, is_training=True)
    
    scaled_attack = transform_scaler(attack_df)
    attack_seq = create_sequences(scaled_attack, SEQUENCE_LENGTH)
    
    predictions, errors = detector.compute_reconstruction_error(attack_seq, per_feature=False), detector.compute_reconstruction_error(attack_seq, per_feature=False)
    predictions = (errors > threshold).astype(int)
    attack_count = sum(predictions)
    print(f"  🚨 Feed Analysis Complete: Found {attack_count} Anomalies out of {len(predictions)} streams.\n")

    # 4. XAI and Semantic Threat Reasoning
    print("🔹 STEP 4 | XAI & SEMANTIC THREAT REASONING")
    print("-" * 65)
    print("  🧠 Triggering Explainable AI Engine...")
    anomaly_idx = next(i for i, p in enumerate(predictions) if p == 1)
    sample_anomaly = attack_seq[anomaly_idx]
    sample_error = errors[anomaly_idx]
    print(f"  🎯 Analyzing Anomaly #{anomaly_idx} (Error: {sample_error:.6f} | Threshold: {threshold:.6f})")

    xai_engine = XAIEngine(detector.model)
    import torch
    seq_tensor = torch.tensor(sample_anomaly, dtype=torch.float32).unsqueeze(0)
    ig_attr = xai_engine.get_integrated_gradients(seq_tensor)

    try:
        xai_engine.plot_ig_heatmap(ig_attr, feature_names)
    except Exception as e:
        print(f"  ⚠️ Heatmap skipped (matplotlib backend issue): {e}")

    top_anomalies = detector.get_most_anomalous_features(sample_anomaly, feature_names, top_k=3)
    xai_engine.semantic_threat_reasoning(top_anomalies)

    # 5. Intelligent Adversary
    print("🔹 STEP 5 | ADVERSARIAL ML TESTING (EVASION ATTACK)")
    print("-" * 65)
    print("  🕵️  Simulating Intelligent Adversary (FGSM)...")
    attacker = AdversarialAttacker(detector.model)
    print(f"  🎯 Target: Force error BELOW threshold ({threshold:.6f})")
    adversarial_seq = attacker.generate_fgsm_attack(sample_anomaly, epsilon=0.015, iterations=30)
    
    adv_errors = detector.compute_reconstruction_error(adversarial_seq, per_feature=False)
    adv_error = adv_errors[0]
    
    print(f"\n  📉 Perturbed Reconstruction Error: {adv_error:.6f}")
    if adv_error < threshold:
        print("  💀 [CRITICAL] EVASION SUCCESS: The Adversary tricked the IDS!")
        print("     The malicious payload is now classified as 'NORMAL' traffic.")
    else:
        print("  🛡️ [SUCCESS] IDS HOLDING: The Adversary failed to spoof the manifold.")

    print("\n" + "=" * 65)
    print("                      DEMONSTRATION COMPLETE                      ")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demonstration()
