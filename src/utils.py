
import matplotlib.pyplot as plt


def plot_errors(errors, threshold):
    plt.figure(figsize=(12, 5))
    plt.plot(errors, label="Reconstruction Error")
    plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
    plt.legend()
    plt.title("Anomaly Detection")
    plt.show()