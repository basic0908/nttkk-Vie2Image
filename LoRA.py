import os
import torch
import datetime
import safetensors.torch
import logging
from typing import Dict
from safetensors import safe_open

# Paths based on your environment
LORA_DIR = r"C:\Users\iizukar\Documents\GitHub\nttkk-Vie2Image\ComfyUI\models\loras"
COND_DIR = r"C:\Users\iizukar\Documents\GitHub\nttkk-Vie2Image\ComfyUI\output\conditioning"
LATENT_DIR = r"C:\Users\iizukar\Documents\GitHub\nttkk-Vie2Image\ComfyUI\output\latents"

class LoRA:
    def __init__(self, rank: int = 128):
        """Initializes a new zeroed LoRA safetensor compatible with ComfyUI."""
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lora_filename = f"lora_{self.timestamp}.safetensors"
        self.lora_path = os.path.join(LORA_DIR, self.lora_filename)
        
        # DreamSync Hyperparameters [cite: 23, 159]
        self.rank = rank 
        self.lr = 0.0001
        self.scheduler = "cosine"
        
        self._create_zero_init_lora()

    def _create_zero_init_lora(self):
        # Path to your ACTUAL model file used in Node 2
        model_path = r"C:\Users\iizukar\Documents\GitHub\nttkk-Vie2Image\ComfyUI\models\unet\flux-2-klein-9b.safetensors"
        state_dict = {}
        
        target_layers = [
            "double_blocks.0.img_attn.qkv",
            "single_blocks.0.attn.qkv"
        ]

        with safe_open(model_path, framework="pt", device="cpu") as f:
            for layer in target_layers:
                weight_key = f"diffusion_model.{layer}.weight"
                if weight_key in f.keys():
                    # Fetch the REAL dimensions from the file
                    real_shape = f.get_slice(weight_key).get_shape()
                    out_features, in_features = real_shape[0], real_shape[1]
                    
                    # Build LoRA keys
                    down_key = f"diffusion_model.{layer}.lora_down.weight"
                    up_key = f"diffusion_model.{layer}.lora_up.weight"
                    
                    state_dict[down_key] = torch.randn(self.rank, in_features) * 0.01
                    state_dict[up_key] = torch.zeros(out_features, self.rank)
                    state_dict[f"diffusion_model.{layer}.alpha"] = torch.tensor(float(self.rank))
                    
                    print(f"Success: Initialized {layer} with In:{in_features}, Out:{out_features}")

        safetensors.torch.save_file(state_dict, self.lora_path)

    def load_training_pair(self, cond_id: str, latent_id: str) -> Dict:
        """Loads conditioning and latent tensors using ComfyUI node logic."""
        cond_path = os.path.join(COND_DIR, f"{cond_id}.cond")
        latent_path = os.path.join(LATENT_DIR, f"{latent_id}.latent")

        # Load Conditioning (weights_only=False for Comfy structures)
        cond_data = torch.load(cond_path, weights_only=False, map_location="cpu")
        conditioning = cond_data["data"]

        # Load Latent using Safetensors
        latent_raw = safetensors.torch.load_file(latent_path, device="cpu")
        
        # Scaling correction: multiplier = 1.0 / 0.18215
        multiplier = 1.0 / 0.18215
        samples = {"samples": latent_raw["latent_tensor"].float() * multiplier}

        print(f"--- Data Loaded for training ---")
        print(f"Conditioning Vector (CLIP/T5): {conditioning[0][0].shape}") 
        print(f"Latent Tensor (VAE): {samples['samples'].shape}")
        
        return {"conditioning": conditioning, "latent": samples}

    def train(self, cond_id: str, latent_id: str):
        data = self.load_training_pair(cond_id, latent_id)
        # Standard iterative bootstrapping loop [cite: 23, 124]
        print(f"Ready to apply iterative alignment training on {self.lora_filename}")

if __name__ == "__main__":
    # Test initialization and data loading
    os.makedirs(COND_DIR, exist_ok=True)
    os.makedirs(LATENT_DIR, exist_ok=True)

    # 1. Create Mock files to test the loaders
    test_cond_path = os.path.join(COND_DIR, "test_001.cond")
    test_latent_path = os.path.join(LATENT_DIR, "test_001.latent")

    # Flux Conditioning (usually batch, tokens, dim) - dim 4096 for T5
    torch.save({"data": [(torch.randn(1, 128, 4096), {"pooled_output": torch.randn(1, 768)})]}, test_cond_path)
    # Flux Latents (batch, channels, h, w) - channels 16 for Flux
    safetensors.torch.save_file({"latent_tensor": torch.randn(1, 16, 128, 128)}, test_latent_path)

    # 2. Run Test
    try:
        worker = LoRA(rank=128)
        # worker.train(cond_id="test_001", latent_id="test_001")
        # print("SUCCESS: LoRA initialized and data pairs mapped.")
    except Exception as e:
        print(f"FAILURE: {e}")