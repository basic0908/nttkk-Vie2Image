import json
import requests
import random
import uuid

def generate_flux_image(workflow, prompt, seed=None, server_address="127.0.0.1:8188"):
    """
    Sends a modified workflow to the ComfyUI API.
    """
    client_id = str(uuid.uuid4())
    
    # 1. Update the Workflow nodes
    # Using .get() or checking keys is safer in case the JSON structure changes
    if "76" in workflow:
        workflow["76"]["inputs"]["value"] = prompt
    else:
        print("Warning: Node 76 not found in workflow.")

    if "75:73" in workflow:
        if seed is None:
            seed = random.randint(1, 10**15)
        workflow["75:73"]["inputs"]["noise_seed"] = seed
    else:
        print("Warning: Node 75:73 not found in workflow.")

    # 2. Create the payload
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }

    # 3. Post to the API
    try:
        url = f"http://{server_address}/prompt"
        # Use json=payload instead of data=json.dumps to automatically set 'Content-Type: application/json'
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print(f"Successfully queued prompt! ID: {result.get('prompt_id')}")
        return result.get('prompt_id')
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to ComfyUI at {server_address}. Is it running?")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Use raw string (r"") for Windows paths to avoid escape character errors
    workflow_file = r"C:/Users/iizukar/Documents/GitHub/nttkk-Vie2Image/flux2_4B-distilled.json"

    try:
        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)
        
        target_prompt = "A rabbit"
        generate_flux_image(workflow_data, target_prompt)
        
    except FileNotFoundError:
        print(f"Error: The file {workflow_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {workflow_file} is not a valid JSON.")