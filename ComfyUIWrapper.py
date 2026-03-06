import json
import requests
import random
import uuid

def generate_flux_image(workflow, prompt, seed=None, server_address="127.0.0.1:8188"):
    """
    Sends a modified workflow to the ComfyUI API.
    
    Args:
        workflow (dict): The loaded JSON workflow.
        prompt (str): The text description to use.
        seed (int): The noise seed. If None, a random one is generated.
        server_address (str): The address of your ComfyUI instance.
    """
    client_id = str(uuid.uuid4())
    
    # Node 76: The text prompt
    workflow["76"]["inputs"]["value"] = prompt
    
    # Node 75:73: The Random Noise seed
    if seed is None:
        seed = random.randint(1, 10**15)
    workflow["75:73"]["inputs"]["noise_seed"] = seed

    # 3. Create the payload
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }

    # 4. Post to the API
    try:
        url = f"http://{server_address}/prompt"
        response = requests.post(url, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        return result['prompt_id']
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to ComfyUI at {server_address}. Is it running?")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None