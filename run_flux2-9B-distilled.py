import json
import urllib.request
import uuid
import random

def queue_prompt(prompt_workflow, server_address="127.0.0.1:8188"):
    p = {"prompt": prompt_workflow, "client_id": str(uuid.uuid4())}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())

# Load workflow JSON
with open("flux2_9B-distilled.json", "r", encoding="utf-8") as f:
    workflow = json.load(f)

for _ in range(5):
    # Insert prompt to the "value" key
    workflow["76"]["inputs"]["value"] = "a green apple on the table"

    # Set random seed
    workflow["75:73"]["inputs"]["noise_seed"] = random.randint(1, 10**15)

    # Send to ComfyUI
    try:
        response = queue_prompt(workflow)
        print(f"Successfully queued! Prompt ID: {response['prompt_id']}")
    except Exception as e:
        print(f"Error connecting to ComfyUI: {e}")