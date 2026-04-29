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
    print(" 🛡️  ROBUST ADV-TRAINING THREAT MODELING DEMONSTRATION  🛡️")
    print("=" * 65 + "\n")

    # 1. Initialize Anomaly Detector & Load Pretrained Model
    print("🔹 STEP 1 | INITIALIZING SYSTEM")
    print("-" * 65)
    print("  ⏳ Loading LSTM Autoencoder (Robust Model)...")
    robust_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "lstm_autoencoder_ADVTraining.pth")
    detector = AnomalyDetector(model_path=robust_model_path)
    print("  ✅ System Initialized.\n")

    # 2. Fit Threshold on Normal Data
    print("🔹 STEP 2 | NORMAL TRAFFIC PROFILING")
    print("-" * 65)
    print("  ⏳ Loading Normal Traffic Profile...")
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

    # 5. Intelligent Adversary & Rigid Defense Evaluation
    print("🔹 STEP 5 | ADVERSARIAL ML TESTING (EVASION ATTACK)")
    print("-" * 65)
    print("  🕵️  Simulating Intelligent Adversary (Iterative Attack)...")
    attacker = AdversarialAttacker(detector.model)
    print(f"  🎯 Target: Force error BELOW threshold ({threshold:.6f})")
    
    # The attacker tries to find a specific value that evades detection
    adversarial_seq = attacker.generate_pgd_attack(sample_anomaly, epsilon=0.015, iterations=30)
    
    # 5a. Evaluation WITH Standard Defense
    adv_errors_std = detector.compute_reconstruction_error(adversarial_seq, per_feature=False, defense_mode=True)
    adv_error_std = adv_errors_std[0]
    
    # 5b. Evaluation WITH Rigid Defense (Randomized Smoothing)
    # This is "nearly rigid" because it averages over 50 noise samples
    print("  🛡️  Applying RIGID Defense (Randomized Smoothing N=50)...")
    adv_errors_rigid = detector.compute_reconstruction_error(adversarial_seq, per_feature=False, defense_mode=True, smoothing_samples=50)
    adv_error_rigid = adv_errors_rigid[0]
    
    print(f"\n  📉 Standard Defense Error: {adv_error_std:.6f}")
    print(f"  🧱 Rigid (Smoothed) Error:   {adv_error_rigid:.6f}")

    if adv_error_rigid > threshold:
        print("  ✅ [RIGID SUCCESS] The Smoothing Defense caught the attacker!")
        print("     Reason: The attacker found a single hole, but the smoothing 'filled' it.")
    elif adv_error_std < threshold:
         print("  💀 [CRITICAL] EVASION SUCCESS: The Adversary tricked the standard IDS!")
    else:
        print("  🛡️ [SUCCESS] IDS HOLDING: The Robust Adversary failed to spoof the manifold.")

    print("\n" + "=" * 65)
    print("                      DEMONSTRATION COMPLETE                      ")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demonstration()
