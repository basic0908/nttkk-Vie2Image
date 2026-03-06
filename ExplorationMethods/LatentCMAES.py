import torch
import numpy as np
import os

class LatentCMAES:
    def __init__(self, latent_dim=None, num_images=8, sigma_init=0.2, learning_rate=0.1, save_path="./"):
        """
        latent_dim: Can be None initially; will be set by set_initial_mean.
        save_path: Directory where latent.csv will be stored.
        """
        self.dim = latent_dim
        self.num_images = num_images
        self.lr = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        
        self.mean = None
        self.sigma = sigma_init
        self.C = None # Initialized once dim is known
        self.history = []
        self.current_candidates = None

    def set_initial_mean(self, base_embedding):
        """Sets the starting point and initializes covariance matrix."""
        if isinstance(base_embedding, np.ndarray):
            base_embedding = torch.from_numpy(base_embedding).to(self.device)
        
        self.mean = base_embedding.float().flatten()
        self.dim = self.mean.shape[0]
        
        # Initialize Covariance Matrix once we know the dimension
        if self.C is None:
            self.C = torch.eye(self.dim, device=self.device)
            
        self._log_latent(self.mean)

    def _log_latent(self, latent_tensor):
        """Moves tensor to CPU and adds to history list."""
        self.history.append(latent_tensor.detach().cpu().numpy().flatten())

    def __call__(self, scores):
        """
        Allows calling flux.expMeth(scores). 
        Calculates the update based on scores and returns the NEW mean.
        """
        if self.current_candidates is None:
            # First run: we haven't generated candidates yet, so we generate them now
            self.current_candidates = self.generate_candidates()
            return self.mean # Return current mean to start

        # Update based on the candidates we showed the user and the EEG scores
        new_mean, _ = self.update(self.current_candidates, scores)
        
        # Generate the NEXT batch of candidates for the next loop
        self.current_candidates = self.generate_candidates()
        
        return new_mean

    def generate_candidates(self):
        """Samples the population from the multivariate normal distribution."""
        dist = torch.distributions.MultivariateNormal(
            self.mean, 
            covariance_matrix=(self.sigma**2) * self.C
        )
        return dist.sample((self.num_images,))

    def update(self, embeddings, scores):
        """The core CMA-ES math: updates mean, covariance, and sigma."""
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
        
        # Normalize weights
        weights = scores / (torch.sum(scores) + 1e-9)
        
        old_mean = self.mean.clone()
        
        # 1. Update Mean (Selection)
        self.mean = torch.sum(weights.unsqueeze(1) * embeddings, dim=0)
        
        # 2. Update Covariance (Rank-1 Update)
        y = (self.mean - old_mean) / self.sigma
        self.C = (1 - self.lr) * self.C + self.lr * torch.ger(y, y)
        
        # 3. Step size (Sigma) control
        move_dist = torch.norm(y)
        self.sigma *= 1.1 if move_dist > 1.0 else 0.9
        
        self._log_latent(self.mean)
        self.to_csv() # Save history to CSV after every update
        
        return self.mean, self.sigma