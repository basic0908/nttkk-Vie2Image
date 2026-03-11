import sys
import os
import json
import requests
import torch
import uuid
import websocket # pip install websocket-client
import urllib.parse

# 1. --- SETUP PATH FIRST ---
base_path = r"C:\Users\ibara\OneDrive\ドキュメント\GitHub\nttkk-Vie2Image"
comfy_path = os.path.join(base_path, "ComfyUI")

# THE FIX: Force priority pathing and change the active directory
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)
os.chdir(comfy_path) # This is the magic line that makes ComfyUI imports work locally!

# 2. --- IMPORT COMFYUI NATIVE CLASSES ---
import comfy.sd      # Good practice to import this first
import folder_paths
import nodes

class FluxImageGen:
    def __init__(self):
        self.base_path = base_path
        self.comfy_input_dir = os.path.join(self.base_path, r"ComfyUI\input")
        
        self.workflow_path = os.path.join(self.base_path, "flux2_4B-distilled_Latent.json")
        
        # Pointing to your EC2 instance!
        self.server_address = "13.193.97.70:8188" 
        self.client_id = str(uuid.uuid4())
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f">>> Using device: {self.device.upper()}")
        
        print(">>> Loading CLIP natively using custom nodes.CLIPLoader()...")
        self.clip_loader = nodes.CLIPLoader()
        self.clip = self.clip_loader.load_clip(
            clip_name="qwen_3_4b.safetensors", 
            type="flux2", 
            device="default"
        )[0]
        self.text_encoder = nodes.CLIPTextEncode()

    def generate_conditioning(self, prompt, filename):
        print(f">>> Encoding prompt: '{prompt}'")
        conditioning = self.text_encoder.encode(self.clip, prompt)[0]
        
        data_to_save = {
            "node_type": "CONDITIONING",
            "data": conditioning
        }
        
        os.makedirs(self.comfy_input_dir, exist_ok=True)
        save_path = os.path.join(self.comfy_input_dir, filename)
        torch.save(data_to_save, save_path)
        print(f">>> Vector saved locally: {save_path}")
        return save_path, filename

    def upload_file_to_server(self, filepath, filename):
        """Uploads the local .cond file to the EC2 server."""
        print(f">>> Uploading {filename} to EC2 server...")
        url = f"http://{self.server_address}/upload/cond"
        
        with open(filepath, "rb") as f:
            files = {"file": (filename, f)}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            print(f"✅ Uploaded {filename} successfully.")
        else:
            print(f"❌ Upload failed: {response.text}")

    def get_image_from_server(self, ws, prompt_id):
        """Listens to WebSocket and downloads the final image."""
        print(">>> Waiting for EC2 to finish generation...")
        
        # 1. Listen to WebSocket until execution finishes
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    # When 'node' is None, the prompt has finished!
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break 
        
        print(">>> Generation complete! Fetching image data...")
        
        # 2. Ask server for the history of our specific prompt
        history_url = f"http://{self.server_address}/history/{prompt_id}"
        history = requests.get(history_url).json()[prompt_id]
        
        # 3. Find the SaveImage node output
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                for image in node_output['images']:
                    filename = image['filename']
                    subfolder = image['subfolder']
                    folder_type = image['type']
                    
                    # 4. Download the actual image bytes
                    query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
                    image_url = f"http://{self.server_address}/view?{query}"
                    
                    image_data = requests.get(image_url).content
                    
                    # 5. Save locally
                    save_path = os.path.join(self.base_path, f"downloaded_{filename}")
                    with open(save_path, "wb") as f:
                        f.write(image_data)
                    print(f"🎉 SUCCESS! Image saved locally to: {save_path}")

    def run_api(self, pos_filename, neg_filename):
        """Sends workflow and starts the WebSocket listener."""
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        workflow["75:74"] = {"inputs": {"conditioning": pos_filename}, "class_type": "LoadConditioning"}
        workflow["75:67"] = {"inputs": {"conditioning": neg_filename}, "class_type": "LoadConditioning"}

        if "75:63" in workflow:
            workflow["75:63"]["inputs"]["positive"] = ["75:74", 0]
            workflow["75:63"]["inputs"]["negative"] = ["75:67", 0]
            workflow["75:63"]["inputs"]["cfg"] = 1.0

        for node in ["76", "75:71", "75:83", "82"]:
            if node in workflow:
                del workflow[node]

        if "75:73" in workflow:
            workflow["75:73"]["inputs"]["noise_seed"] = torch.randint(0, 10**15, (1,)).item()

        # Connect WebSocket BEFORE triggering prompt so we don't miss messages
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")

        payload = {"prompt": workflow, "client_id": self.client_id}
        url = f"http://{self.server_address}/prompt"
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            pid = response.json().get("prompt_id")
            print(f"✅ EC2 API Queued. ID: {pid}")
            
            # Start listening for the finished image
            self.get_image_from_server(ws, pid)
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    gen = FluxImageGen()
    
    # 1. Generate Local Vectors
    pos_path, pos_name = gen.generate_conditioning(
        "a yellow pikming warrior with a sword, in the style of Studio Ghibli", 
        "ComfyUI_positive.cond"
    )
    neg_path, neg_name = gen.generate_conditioning("", "ComfyUI_negative.cond")
    
    # 2. Upload Vectors to EC2
    gen.upload_file_to_server(pos_path, pos_name)
    gen.upload_file_to_server(neg_path, neg_name)
    
    # 3. Fire API and wait for Image over WebSockets
    gen.run_api(pos_name, neg_name)