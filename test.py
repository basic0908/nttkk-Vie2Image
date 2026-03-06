import os
import torch
import safetensors.torch
from safetensors import safe_open

# Update these paths to your local environment
MODEL_PATH = r"C:\Users\iizukar\Documents\GitHub\nttkk-Vie2Image\ComfyUI\models\diffusion_models\flux-2-klein-4b.safetensors"
LORA_OUTPUT_DIR = r"C:\Users\iizukar\Documents\GitHub\nttkk-Vie2Image\ComfyUI\models\loras"

def create_flux_4b_lora(model_path, output_dir, rank=32):
    """
    Probes the 4B model and creates a compatible, zero-initialized LoRA.
    Using Rank 32 to ensure stability with noisy EEG signals.
    """
    os.makedirs(output_dir, exist_ok=True)
    state_dict = {}
    
    # These strings target the main attention blocks in Flux transformers
    target_subsets = ["img_attn.qkv", "txt_attn.qkv", "attn.qkv"]
    
    print(f"Probing model: {os.path.basename(model_path)}")
    
    try:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # We target weights in the double and single blocks
                if any(sub in key for sub in target_subsets) and "weight" in key:
                    # Get the actual shape from the 4B model
                    # Flux weights are [Out, In]
                    shape = f.get_slice(key).get_shape()
                    out_dim, in_dim = shape[0], shape[1]
                    
                    # Clean the key for LoRA naming (remove '.weight')
                    base_name = key.replace(".weight", "")
                    
                    # ComfyUI-style LoRA keys
                    down_key = f"{base_name}.lora_down.weight"
                    up_key = f"{base_name}.lora_up.weight"
                    alpha_key = f"{base_name}.alpha"
                    
                    # Initialization
                    # Down: Random (scaled) | Up: Zeros (Identity)
                    state_dict[down_key] = torch.randn(rank, in_dim) * 0.01
                    state_dict[up_key] = torch.zeros(out_dim, rank)
                    state_dict[alpha_key] = torch.tensor(float(rank))
                    
                    print(f"Mapped {base_name}: [{in_dim} -> {out_dim}]")

        # Save as Safetensor
        output_path = os.path.join(output_dir, "flux2_4b_eeg_alignment.safetensors")
        safetensors.torch.save_file(state_dict, output_path)
        print(f"\n✅ Successfully created LoRA for 4B Model: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during probing: {e}")

if __name__ == "__main__":
    # Recommend Rank 32 or 64 for EEG alignment to prevent overfitting to noise
    create_flux_4b_lora(MODEL_PATH, LORA_OUTPUT_DIR, rank=64)