import os
import json
import requests
import uuid
import websocket
import urllib.parse
import torch

# --- CONFIGURATION ---
BASE_PATH = r"C:\Users\ibara\OneDrive\ドキュメント\GitHub\nttkk-Vie2Image"
SERVER_ADDRESS = "13.193.97.70:8188" # Your Elastic IP
WORKFLOW_JSON = os.path.join(BASE_PATH, "flux2_4B-distilled_Latent.json")
INPUT_DIR = os.path.join(BASE_PATH, "ComfyUI", "input")
CLIENT_ID = str(uuid.uuid4())

def upload_to_ec2(filename):
    url = f"http://{SERVER_ADDRESS}/upload/cond"
    path = os.path.join(INPUT_DIR, filename)
    with open(path, "rb") as f:
        files = {"file": (filename, f)}
        r = requests.post(url, files=files)
    if r.status_code == 200:
        print(f"✅ {filename} uploaded.")
    else:
        print(f"❌ Upload failed: {r.text}")

def run_flux_on_ec2(pos_name, neg_name):
    with open(WORKFLOW_JSON, "r", encoding="utf-8") as f:
        wf = json.load(f)

    # Map the uploaded vectors to the Load nodes
    wf["75:74"] = {"inputs": {"conditioning": pos_name}, "class_type": "LoadConditioning"}
    wf["75:67"] = {"inputs": {"conditioning": neg_name}, "class_type": "LoadConditioning"}
    
    # Randomize Seed
    if "75:73" in wf:
        wf["75:73"]["inputs"]["noise_seed"] = torch.randint(0, 10**15, (1,)).item()

    # WebSocket setup
    ws = websocket.WebSocket()
    ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={CLIENT_ID}")

    # Trigger Prompt
    print(">>> Triggering EC2 Generation...")
    resp = requests.post(f"http://{SERVER_ADDRESS}/prompt", json={"prompt": wf, "client_id": CLIENT_ID})
    
    if resp.status_code == 200:
        pid = resp.json().get("prompt_id")
        print(f"✅ Queued: {pid}. Waiting for WebSocket...")
        
        # Simple loop to wait for finish
        while True:
            msg = ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data['type'] == 'executing' and data['data']['node'] is None:
                    break
        
        print("🎉 Done! You can now fetch your image from history or /view.")
    else:
        print(f"❌ Error: {resp.text}")

if __name__ == "__main__":
    upload_to_ec2("pos.cond")
    upload_to_ec2("neg.cond")
    run_flux_on_ec2("pos.cond", "neg.cond")