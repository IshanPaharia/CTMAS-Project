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
    df = load_data("../data/normal.csv")

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

    print("Training...")
    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in loader:
            batch = batch.to(DEVICE)

            output = model(batch)
            loss = criterion(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")


if __name__ == "__main__":
    train()