import os
import json
import requests
import uuid
import websocket # Ensure you have websocket-client installed, not websocket
import urllib.parse
import torch

# --- CONFIGURATION ---
BASE_PATH = r"C:\Users\ibara\OneDrive\ドキュメント\GitHub\nttkk-Vie2Image"

SERVER_ADDRESS = "13.193.97.70:8188" # Your Elastic IP
WORKFLOW_JSON = os.path.join(BASE_PATH, "flux2_4B-distilled_Latent.json")
INPUT_DIR = os.path.join(BASE_PATH, "ComfyUI", "input")
SAVE_DIR = os.path.join(BASE_PATH, "images")
CLIENT_ID = str(uuid.uuid4())

def upload_to_ec2(filename):
    """Uploads the local .cond file to the EC2 server."""
    print(f"Now uploading: {filename} to EC2...")
    url = f"http://{SERVER_ADDRESS}/upload/cond"
    path = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found. Run vector_generator.py first.")
        return False

    with open(path, "rb") as f:
        files = {"file": (filename, f)}
        r = requests.post(url, files=files)
    
    if r.status_code == 200:
        print(f"✅ {filename} uploaded successfully.")
        return True
    else:
        print(f"❌ Upload failed: {r.text}")
        return False

def run_flux_on_ec2(pos_name, neg_name):
    """Updates workflow and triggers generation via WebSocket."""
    with open(WORKFLOW_JSON, "r", encoding="utf-8") as f:
        wf = json.load(f)

    # --- 0. PATH SANITIZATION (The Fix for FileNotFoundError) ---
    # This replaces Windows paths in the JSON with the Linux path you created
    linux_csv_path = "/home/ec2-user/nttkk-Vie2Image/output/csv"
    for node_id in wf:
        if "inputs" in wf[node_id]:
            for key, value in wf[node_id]["inputs"].items():
                if isinstance(value, str) and "C:\\Users\\" in value:
                    wf[node_id]["inputs"][key] = linux_csv_path
                    print(f"🔧 Corrected path in node {node_id}: {key}")

    # 1. FIX CONDITIONING FILENAMES
    wf["75:74"] = {
        "class_type": "LoadConditioning",
        "inputs": {"conditioning": pos_name}
    }
    wf["75:83"] = {
        "class_type": "LoadConditioning",
        "inputs": {"conditioning": neg_name}
    }

    # 2. ENSURE MODELS ARE MAPPED CORRECTLY
    if "75:72" in wf:
        wf["75:72"]["inputs"]["vae_name"] = "flux2-vae.safetensors"
    if "75:81" in wf:
        wf["75:81"]["inputs"]["unet_name"] = "flux-2-klein-4b.safetensors"

    # 3. FIX CFG GUIDER CONNECTIONS
    if "75:63" in wf:
        wf["75:63"]["inputs"]["positive"] = ["75:74", 0]
        wf["75:63"]["inputs"]["negative"] = ["75:83", 0]
        wf["75:63"]["inputs"]["cfg"] = 1.0

    # 4. RANDOMIZE SEED
    if "75:73" in wf:
        wf["75:73"]["inputs"]["noise_seed"] = torch.randint(0, 10**15, (1,)).item()

    # 5. CONNECT WEBSOCKET
    print(">>> Connecting to WebSocket...")
    ws = websocket.create_connection(f"ws://{SERVER_ADDRESS}/ws?clientId={CLIENT_ID}")

    # 6. TRIGGER PROMPT
    print(">>> Triggering EC2 Generation...")
    payload = {"prompt": wf, "client_id": CLIENT_ID}
    resp = requests.post(f"http://{SERVER_ADDRESS}/prompt", json=payload)
    
    if resp.status_code == 200:
        prompt_id = resp.json().get("prompt_id")
        print(f"✅ Queued Prompt ID: {prompt_id}")
        
        # 7. WAIT FOR COMPLETION
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        print("🎉 Execution Finished!")
                        break
        
        # 8. FETCH AND DOWNLOAD THE IMAGE
        fetch_and_save_image(prompt_id)
    else:
        print(f"❌ API Validation Error: {resp.text}")

def fetch_and_save_image(prompt_id):
    """Asks the server for the result and downloads the image."""
    history_url = f"http://{SERVER_ADDRESS}/history/{prompt_id}"
    history = requests.get(history_url).json()[prompt_id]
    
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'images' in node_output:
            for image in node_output['images']:
                filename = image['filename']
                query = urllib.parse.urlencode({
                    "filename": filename, 
                    "subfolder": image['subfolder'], 
                    "type": image['type']
                })
                image_url = f"http://{SERVER_ADDRESS}/view?{query}"
                
                print(f">>> Downloading {filename}...")
                img_data = requests.get(image_url).content
                save_path = os.path.join(SAVE_DIR, f"result_{filename}")
                
                with open(save_path, "wb") as f:
                    f.write(img_data)
                print(f"⭐ Image saved locally: {save_path}")

if __name__ == "__main__":
    # Ensure vectors are uploaded before running
    if upload_to_ec2("pos.cond"):
        run_flux_on_ec2("pos.cond", "neg.cond")