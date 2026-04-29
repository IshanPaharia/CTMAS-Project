import torch
from config import DEVICE

class AdversarialAttacker:
    def __init__(self, model):
        """
        Initializes a PGD-based Intelligent Adversary.
        Purpose: Evasion Attack. Attempts to perturb an Attack sample
        such that the Reconstruction Error drops below the threshold,
        tricking the system into classifying it as "Normal".
        """
        self.model = model.to(DEVICE)
        self.model.eval()
        self.device = DEVICE

    def generate_pgd_attack(self, sequence, epsilon=0.01, iterations=10):
        """
        Generates an adversarial evasion sequence using Projected Gradient Descent (PGD).
        The adversary's goal is to minimize the reconstruction loss!
        """
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).to(self.device).clone()
        if len(seq_tensor.shape) == 2:
            seq_tensor = seq_tensor.unsqueeze(0)
            
        adv_sequence = seq_tensor.clone().detach().requires_grad_(True)
        
        for i in range(iterations):
            adv_sequence.requires_grad = True
            
            recon = self.model(adv_sequence)
            
            # Loss is MSE reconstruction error
            loss = ((adv_sequence - recon) ** 2).mean()
            
            self.model.zero_grad()
            loss.backward()
            
            # The adversary minimizes the loss to evade detection
            gradient = adv_sequence.grad.data
            
            # Step in the direction of negative gradient to minimize loss
            adv_sequence = adv_sequence - epsilon * torch.sign(gradient)
            
            # Clamp values to valid normalized feature ranges [0, 1] if input was MinMax scaled
            adv_sequence = torch.clamp(adv_sequence, 0.0, 1.0).detach()
            
        return adv_sequence.cpu().numpy()
