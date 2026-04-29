import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from config import *
from preprocessing import load_data, fit_scaler, create_sequences
from dataset import TimeSeriesDataset
from model import LSTMAutoencoder


def train():
    os.makedirs("models", exist_ok=True)

    print("Loading NORMAL data...")
    df, _, _ = load_data(NORMAL_DATA_PATH, is_training=True)

    print("Scaling...")
    scaled_data, scaler = fit_scaler(df)

    print("Creating sequences...")
    sequences = create_sequences(scaled_data, SEQUENCE_LENGTH)

    dataset = TimeSeriesDataset(sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    n_features = sequences.shape[2]

    model = LSTMAutoencoder(n_features).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    print("Training Adversarially (Robust Mode)...")
    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in loader:
            batch = batch.to(DEVICE)

            # --- 1. Adversarial Generation (PGD - Multi-step) ---
            # PGD is much more "rigid" than FGSM because it searches the space better
            batch_adv = batch.clone().detach()
            
            # Hyperparameters for PGD
            epsilon = 0.03   # Max perturbation
            alpha = 0.01     # Step size
            iters = 7        # Number of steps
            
            # Start from a random point within epsilon for better coverage
            batch_adv = batch_adv + torch.empty_like(batch_adv).uniform_(-epsilon, epsilon)
            batch_adv = torch.clamp(batch_adv, 0.0, 1.0).detach()

            for _ in range(iters):
                batch_adv.requires_grad = True
                output_temp = model(batch_adv)
                loss_temp = criterion(output_temp, batch)
                
                model.zero_grad()
                loss_temp.backward()
                
                # Maximizing reconstruction loss (Attacker's goal in training is to be strong)
                with torch.no_grad():
                    adv_grad = batch_adv.grad.sign()
                    batch_adv = batch_adv + alpha * adv_grad
                    
                    # Project back to epsilon ball and valid range
                    eta = torch.clamp(batch_adv - batch, -epsilon, epsilon)
                    batch_adv = torch.clamp(batch + eta, 0.0, 1.0).detach()
            
            # --- 2. Robust Training (Clean + PGD Noisy) ---
            optimizer.zero_grad()
            
            # Clean loss
            output_clean = model(batch)
            clean_loss = criterion(output_clean, batch)
            
            # Robust loss
            output_robust = model(batch_adv)
            robust_loss = criterion(output_robust, batch)
            
            # Combined Loss (Weighting robust loss higher helps rigidity)
            total_batch_loss = clean_loss + 1.5 * robust_loss
            
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.6f}")

    target_path = os.path.join("models", "lstm_autoencoder_ADVTraining.pth")
    torch.save(model.state_dict(), target_path)
    print(f"Robust Model saved to {target_path}!")


if __name__ == "__main__":
    train()
