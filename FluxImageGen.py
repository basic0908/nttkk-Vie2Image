import sys
import os
import json
import requests
import torch
import uuid

# 1. --- SETUP PATH FIRST ---
base_path = r"C:\Users\iizukar\Documents\GitHub\nttkk-Vie2Image"
comfy_path = os.path.join(base_path, "ComfyUI")
if comfy_path not in sys.path:
    sys.path.append(comfy_path)

# 2. --- IMPORT COMFYUI NATIVE CLASSES ---
import folder_paths
import nodes

class FluxImageGen:
    def __init__(self):
        self.base_path = base_path
        self.comfy_input_dir = os.path.join(self.base_path, r"ComfyUI\input")
        
        self.workflow_path = "flux2_4B-distilled_Latent.json"
        self.server_address = "127.0.0.1:8188"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f">>> Using device: {self.device.upper()}")
        
        print(">>> Loading CLIP natively using custom nodes.CLIPLoader()...")
        
        # THE FIX: Route through your custom node so "flux2" gets uppercase'd and padded to 3072!
        self.clip_loader = nodes.CLIPLoader()
        self.clip = self.clip_loader.load_clip(
            clip_name="qwen_3_4b.safetensors", 
            type="flux2", 
            device="default"
        )[0]
        
        self.text_encoder = nodes.CLIPTextEncode()

    def generate_conditioning(self, prompt, filename):
        print(f">>> Encoding prompt: '{prompt}'")
        
        # 1. Get exact padded tensor structure
        conditioning = self.text_encoder.encode(self.clip, prompt)[0]
        
        # 2. Wrap perfectly for LoadConditioning
        data_to_save = {
            "node_type": "CONDITIONING",
            "data": conditioning
        }
        
        os.makedirs(self.comfy_input_dir, exist_ok=True)
        save_path = os.path.join(self.comfy_input_dir, filename)
        
        torch.save(data_to_save, save_path)
        
        print(f">>> Vector saved successfully: {save_path}")
        return filename

    def run_api(self, pos_cond_file, neg_cond_file):
        """Sends workflow to ComfyUI."""
        if not os.path.exists(self.workflow_path):
            print(f"❌ Workflow file {self.workflow_path} not found.")
            return None

        with open(self.workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # --- Inject Positive Vector ---
        workflow["75:74"] = {
            "inputs": {
                "conditioning": pos_cond_file
            },
            "class_type": "LoadConditioning"
        }

        # --- Inject Negative Vector ---
        workflow["75:67"] = {
            "inputs": {
                "conditioning": neg_cond_file
            },
            "class_type": "LoadConditioning"
        }

        # --- Satisfy CFGGuider Graph Connections ---
        if "75:63" in workflow:
            workflow["75:63"]["inputs"]["positive"] = ["75:74", 0]
            workflow["75:63"]["inputs"]["negative"] = ["75:67", 0]
            workflow["75:63"]["inputs"]["cfg"] = 1.0

        # --- Cleanup Unused Nodes ---
        nodes_to_remove = ["76", "75:71", "75:83", "82"]
        for node in nodes_to_remove:
            if node in workflow:
                del workflow[node]

        # Randomize Seed
        if "75:73" in workflow:
            workflow["75:73"]["inputs"]["noise_seed"] = torch.randint(0, 10**15, (1,)).item()

        payload = {
            "prompt": workflow,
            "client_id": str(uuid.uuid4())
        }

        try:
            url = f"http://{self.server_address}/prompt"
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                pid = response.json().get("prompt_id")
                print(f"✅ Success! Image Generation Queued. ID: {pid}")
                return pid
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    gen = FluxImageGen()
    
    pos_file = gen.generate_conditioning(
        "a little girl wearing a bright yellow dress and a copper crown is riding a badger through a field of flowers", 
        "ComfyUI_positive.cond"
    )
    
    neg_file = gen.generate_conditioning(
        "", 
        "ComfyUI_negative.cond"
    )
    
    gen.run_api(pos_file, neg_file)