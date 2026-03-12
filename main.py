import os
import sys
import json
import requests
import uuid
import websocket
import urllib.parse
import torch
import msvcrt  # Native Windows module to capture Esc/Enter keys
import time

# --- IMPORT YOUR LOCAL VECTOR GENERATOR ---
import vector_generator 

# --- EXPERIMENT CONFIGURATION ---
BASE_PATH = r"C:\Users\ibara\OneDrive\ドキュメント\GitHub\nttkk-Vie2Image"
SERVER_ADDRESS = "13.193.97.70:8188" 
WORKFLOW_JSON = os.path.join(BASE_PATH, "flux2_4B-distilled_Latent.json")
INPUT_DIR = os.path.join(BASE_PATH, "ComfyUI", "input")
SAVE_DIR = os.path.join(BASE_PATH, "images")

# The number of images to generate per Enter press
NUM_IMAGES_PER_TRIAL = 8 

CLIENT_ID = str(uuid.uuid4())

def upload_to_ec2(filepath, filename):
    """Uploads a specific .cond file to the EC2 server."""
    print(f"☁️ Uploading {filename} to EC2...")
    url = f"http://{SERVER_ADDRESS}/upload/cond"
    
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found.")
        return False

    with open(filepath, "rb") as f:
        files = {"file": (filename, f)}
        r = requests.post(url, files=files)
    
    if r.status_code == 200:
        return True
    else:
        print(f"❌ Upload failed: {r.text}")
        return False

def run_generation(pos_name, subject_dir, trial_num, batch_size):
    """Triggers generation for a batch of images via WebSocket."""
    with open(WORKFLOW_JSON, "r", encoding="utf-8") as f:
        wf = json.load(f)

    # 1. LOAD CONDITIONING 
    # Positive (Uploaded dynamically)
    wf["75:83"] = {
        "class_type": "LoadConditioning", 
        "inputs": {"conditioning": pos_name}
    }
    # Negative (Uses your persistent neg.cond on the EC2 to satisfy validation)
    wf["999:99"] = { 
        "class_type": "LoadConditioning", 
        "inputs": {"conditioning": "neg.cond"}
    }

    # 2. SET MODELS
    if "75:72" in wf: wf["75:72"]["inputs"]["vae_name"] = "flux2-vae.safetensors"
    if "75:81" in wf: wf["75:81"]["inputs"]["unet_name"] = "flux-2-klein-4b.safetensors"

    # 3. SET CFG GUIDER
    if "75:63" in wf:
        wf["75:63"]["inputs"]["positive"] = ["75:83", 0]
        wf["75:63"]["inputs"]["negative"] = ["999:99", 0] # Reconnected to satisfy validation
        wf["75:63"]["inputs"]["cfg"] = 1.0

    # 4. RANDOMIZE SEED 
    if "75:73" in wf:
        wf["75:73"]["inputs"]["noise_seed"] = torch.randint(0, 10**15, (1,)).item()

    # 5. SET BATCH SIZE (Targeting EmptyFlux2LatentImage directly)
    if "75:66" in wf:
        wf["75:66"]["inputs"]["batch_size"] = batch_size
    else:
        # Fallback search just in case the ID changes
        for node_id, node_info in wf.items():
            if node_info.get("class_type") in ["EmptyLatentImage", "EmptyFlux2LatentImage"]:
                node_info["inputs"]["batch_size"] = batch_size

    # 6. TRIGGER AND WAIT
    ws = websocket.create_connection(f"ws://{SERVER_ADDRESS}/ws?clientId={CLIENT_ID}")
    resp = requests.post(f"http://{SERVER_ADDRESS}/prompt", json={"prompt": wf, "client_id": CLIENT_ID})
    
    if resp.status_code == 200:
        prompt_id = resp.json().get("prompt_id")
        
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break
        
        ws.close()
        fetch_and_save_images(prompt_id, subject_dir, trial_num)
    else:
        print(f"❌ API Error: {resp.text}")

def fetch_and_save_images(prompt_id, subject_dir, trial_num):
    """Downloads all generated images in the batch and names them logically."""
    history_url = f"http://{SERVER_ADDRESS}/history/{prompt_id}"
    history = requests.get(history_url).json()[prompt_id]
    
    img_counter = 1 
    
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'images' in node_output:
            for image in node_output['images']:
                filename = image['filename']
                query = urllib.parse.urlencode({"filename": filename, "subfolder": image['subfolder'], "type": image['type']})
                image_url = f"http://{SERVER_ADDRESS}/view?{query}"
                
                img_data = requests.get(image_url).content
                
                if trial_num == 0:
                    save_name = f"Warmup_Img{img_counter:02d}.png"
                else:
                    save_name = f"Trial{trial_num:02d}_Img{img_counter:02d}.png"
                
                save_path = os.path.join(subject_dir, save_name)
                
                with open(save_path, "wb") as f:
                    f.write(img_data)
                print(f"⭐ Saved: {save_path}")
                img_counter += 1

def run_experiment():
    print("="*50)
    print("🧠 Vie2Image EEG Experiment Controller")
    print("="*50)
    
    subject_name = input("\nEnter Subject Name (e.g., Subj01): ").strip()
    prompt = input("Enter Prompt for this session: ").strip()

    # Create Subject Folder
    subject_dir = os.path.join(SAVE_DIR, subject_name)
    os.makedirs(subject_dir, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Save Prompt Text
    with open(os.path.join(subject_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    # Generate Local Vector
    print("\n>>> Generating Positive Conditioning Vector Locally...")
    cond_filename = "pos.cond"
    cond_filepath = os.path.join(INPUT_DIR, cond_filename)
    
    vector_generator.generate_local_vectors(prompt, cond_filepath) 

    # Upload to EC2
    upload_to_ec2(cond_filepath, cond_filename)

    # WARM UP (Trial 0) - Generate 1 image to load models into VRAM
    print("\n>>> Warming up EC2 Pipeline (Testing connection and loading models to VRAM)...")
    run_generation(cond_filename, subject_dir, trial_num=0, batch_size=1)
    print("✅ Warm-up complete. Models are loaded in VRAM.")

    # EXPERIMENT LOOP
    trial_count = 1
    while True:
        print(f"\n--- Ready for Trial {trial_count} ---")
        print(f"Press [ENTER] to generate a batch of {NUM_IMAGES_PER_TRIAL} images.")
        print(f"Press [ESC] to exit experiment.")
        
        while True:
            key = msvcrt.getch()
            if key in (b'\r', b'\n'): # Enter Key
                break
            elif key == b'\x1b': # Esc Key
                print("\n🛑 Exiting Experiment. Goodbye!")
                sys.exit()

        print(f"\n🚀 Processing Batch of {NUM_IMAGES_PER_TRIAL} for Trial {trial_count}...")
        
        run_generation(cond_filename, subject_dir, trial_count, batch_size=NUM_IMAGES_PER_TRIAL)
            
        print(f"✅ Trial {trial_count} complete!")
        trial_count += 1

if __name__ == "__main__":
    run_experiment()