import torch
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
import os

def create_flux_dreamsync_lora():
    # --- CONFIGURATION FROM DREAMSYNC TABLE 7 ---
    rank = 128  # 
    
    flux_config = {
        "attention_head_dim": 128,
        "in_channels": 64,
        "joint_attention_dim": 4096,
        "num_attention_heads": 24,
        "num_layers": 19,         # Double blocks [cite: 808]
        "num_single_layers": 38,  # Single blocks [cite: 808]
        "patch_size": 1,
        "pooled_projection_dim": 768,
        "guidance_embeds": True,
        "_class_name": "FluxTransformer2DModel"
    }

    print(f"Creating DreamSync-spec LoRA (Rank {rank})...")
    transformer = FluxTransformer2DModel.from_config(flux_config)
    transformer.requires_grad_(False)

    # Targeting attention projections [cite: 23, 156]
    flux_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank, 
        init_lora_weights="gaussian", # Matrix B=ZERO [cite: 156]
        target_modules=flux_target_modules,
    )

    peft_model = get_peft_model(transformer, lora_config)
    state_dict = peft_model.state_dict()
    
    final_dict = {}
    for key, value in state_dict.items():
        if "lora" not in key:
            continue

        # 1. Clean PEFT prefixes
        new_key = key.replace("base_model.model.", "")
        
        # 2. Standardize LoRA mapping for ComfyUI
        new_key = new_key.replace("lora_A.weight", "lora_down.weight")
        new_key = new_key.replace("lora_B.weight", "lora_up.weight")

        # 3. Handle the Split Architecture Naming
        # Maps transformer_blocks (0-18) to double_blocks
        # Maps single_transformer_blocks (0-37) to single_blocks
        if "transformer_blocks" in new_key:
            new_key = new_key.replace("transformer_blocks", "double_blocks")
        elif "single_transformer_blocks" in new_key:
            new_key = new_key.replace("single_transformer_blocks", "single_blocks")

        final_dict[new_key] = value.to(torch.float16)

    return final_dict

if __name__ == "__main__":
    # Update to your actual path
    COMFY_LORA_PATH = r"C:\Path\To\ComfyUI\models\loras\flux_dreamsync_init.safetensors"
    
    try:
        tensors = create_flux_dreamsync_lora()
        os.makedirs(os.path.dirname(COMFY_LORA_PATH), exist_ok=True)
        save_file(tensors, COMFY_LORA_PATH)
        print(f"✅ Created Rank {128} LoRA compatible with Flux Single/Double blocks.")
    except Exception as e:
        print(f"❌ Error: {e}")