import json
import urllib.request
import uuid
import random
import argparse
import os
from datetime import datetime

def queue_prompt(prompt_workflow, server_address="127.0.0.1:8188"):
    p = {"prompt": prompt_workflow, "client_id": str(uuid.uuid4())}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())

def main():
    parser = argparse.ArgumentParser(description="Run Flux2 Klein Inference")
    parser.add_argument("--prompt", type=str, 
                        default="a little girl wearing a bright yellow dress and a copper crown is riding a badger through a field of flowers")
    parser.add_argument("--params", type=str, choices=["4B", "9B"], default="9B")
    parser.add_argument("--batch", type=int, default=4)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--output_dir", type=str, default=f"Flux2-Klein-{{params}}-{timestamp}")

    args = parser.parse_args()

    # Determine which JSON file to load and which node IDs to target
    workflow_file = "flux2_9B-distilled.json" if args.params == "9B" else "flux2_4B-distilled.json"
    unet_node_id = "75:85" if args.params == "9B" else "75:81"
    clip_node_id = "75:71"
    
    if not os.path.exists(workflow_file):
        print(f"Error: {workflow_file} not found in current directory.")
        return

    with open(workflow_file, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Setup parameters for the specific model version
    if args.params == "4B":
        workflow[unet_node_id]["inputs"]["unet_name"] = "flux-2-klein-4b.safetensors"
        workflow[clip_node_id]["inputs"]["clip_name"] = "qwen_3_4b.safetensors"
    else:
        workflow[unet_node_id]["inputs"]["unet_name"] = "flux-2-klein-9b.safetensors"
        workflow[clip_node_id]["inputs"]["clip_name"] = "qwen_3_8b_fp8mixed.safetensors"

    # Format output prefix
    final_output_path = args.output_dir.replace("{params}", args.params)
    workflow["9"]["inputs"]["filename_prefix"] = f"{final_output_path}/gen"

    print(f"--- Running {args.params} Workflow ---")
    print(f"Prompt: {args.prompt}")

    for i in range(args.batch):
        # Update Prompt and Seed (IDs remain consistent across your JSONs)
        workflow["76"]["inputs"]["value"] = args.prompt
        workflow["75:73"]["inputs"]["noise_seed"] = random.randint(1, 10**15)

        try:
            res = queue_prompt(workflow)
            print(f"[{i+1}/{args.batch}] Queued! ID: {res['prompt_id']}")
        except Exception as e:
            print(f"Connection Failed: {e}")

if __name__ == "__main__":
    main()