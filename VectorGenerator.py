import sys
import os
import torch
import nodes # Requires ComfyUI in your PYTHONPATH

# --- SETUP PATHS ---
base_path = r"C:\Users\ibara\OneDrive\ドキュメント\GitHub\nttkk-Vie2Image"
comfy_path = os.path.join(base_path, "ComfyUI")

if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)
os.chdir(comfy_path)

def generate_local_vectors(prompt_text, neg_text=""):
    print(">>> Initializing local CLIP (Qwen 3 4B)...")
    
    # Load CLIP natively
    clip_loader = nodes.CLIPLoader()
    clip = clip_loader.load_clip(
        clip_name="qwen_3_4b.safetensors", 
        type="flux2", 
        device="default"
    )[0]
    
    text_encoder = nodes.CLIPTextEncode()
    input_dir = os.path.join(base_path, "ComfyUI", "input")
    os.makedirs(input_dir, exist_ok=True)

    # Encode Positive
    print(f">>> Encoding Positive: '{prompt_text}'")
    pos_cond = text_encoder.encode(clip, prompt_text)[0]
    torch.save({"node_type": "CONDITIONING", "data": pos_cond}, 
               os.path.join(input_dir, "pos.cond"))

    # Encode Negative
    print(f">>> Encoding Negative: '{neg_text}'")
    neg_cond = text_encoder.encode(clip, neg_text)[0]
    torch.save({"node_type": "CONDITIONING", "data": neg_cond}, 
               os.path.join(input_dir, "neg.cond"))

    print(f"✅ Vectors saved to {input_dir}")

if __name__ == "__main__":
    p = "a yellow pikming warrior with a sword, in the style of Studio Ghibli"
    generate_local_vectors(p)