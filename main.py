import json
import argparse
import os
from datetime import datetime
import uuid
import random
import requests
import sys
import keyboard
import numpy as np


class FluxInference:

    def __init__(self):
        self.args = self.parse_args()

        self.ExplorationMethods = ["CMAES", "Bayesian", "Slerp"]

        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Workflow file selection
        self.workflow_file = (
            "flux2_9B-distilled.json"
            if self.args.params == "9B"
            else "flux2_4B-distilled.json"
        )

        self.unet_node_id = "75:85" if self.args.params == "9B" else "75:81"
        self.clip_node_id = "75:71"

        if not os.path.exists(self.workflow_file):
            raise FileNotFoundError(self.workflow_file)

        with open(self.workflow_file, "r", encoding="utf-8") as f:
            self.workflow_template = json.load(f)

        self.workflow_file_latent = (
            "flux2_9B-distilled_latent.json"
            if self.args.params == "9B"
            else "flux2_4B-distilled_latent.json"
        )

        self.unet_node_id = "75:85" if self.args.params == "9B" else "75:81"
        self.clip_node_id = "75:71"

        if not os.path.exists(self.workflow_file_latent):
            raise FileNotFoundError(self.workflow_file_latent)

        with open(self.workflow_file_latent, "r", encoding="utf-8") as f:
            self.workflow_template_latent = json.load(f)

        # Inject model names
        if self.args.params == "4B":
            self.workflow_template[self.unet_node_id]["inputs"]["unet_name"] = "flux-2-klein-4b.safetensors"
            self.workflow_template[self.clip_node_id]["inputs"]["clip_name"] = "qwen_3_4b.safetensors"
            self.workflow_template_latent[self.unet_node_id]["inputs"]["unet_name"] = "flux-2-klein-4b.safetensors"
            self.workflow_template_latent[self.clip_node_id]["inputs"]["clip_name"] = "qwen_3_4b.safetensors"
        else:
            self.workflow_template[self.unet_node_id]["inputs"]["unet_name"] = "flux-2-klein-9b.safetensors"
            self.workflow_template[self.clip_node_id]["inputs"]["clip_name"] = "qwen_3_8b_fp8mixed.safetensors"

        # Cache node references (faster than dict lookup each call)
        self.prompt_node = "76"
        self.seed_node = "75:73"

        # Network
        self.server_address = "127.0.0.1:8188"
        self.url = f"http://{self.server_address}/prompt"
        self.session = requests.Session()

        # Random generator
        self.rng = random.Random()

        print(f"--- Running {self.args.params} Workflow ---")
        print(f"Exploration Method: {self.ExplorationMethods[self.args.ExplorationMethod - 1]}")
        print(f"Prompt: {self.args.prompt}")
        print(f"Batch Size: {self.args.batch}")

        # Latent settings
        self.expMeth = None

        # Output directories
        self.image_path = os.path.join(os.getcwd(), "output/images", self.timestamp)
        self.csv_path = os.path.join(os.getcwd(), "output/csv", self.timestamp)

        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.csv_path, exist_ok=True)

        try:
            if self.args.ExplorationMethod == 1:
                from ExplorationMethods.LatentCMAES import LatentCMAES
                self.expMeth = LatentCMAES(
                    latent_dim=None,
                    num_images=self.args.batch,
                    sigma_init=0.2,
                    learning_rate=0.1,
                    save_path=self.csv_path  # Pass the timestamped path here
                )

            elif self.args.ExplorationMethod == 2:
                from ExplorationMethods.Bayesian import Bayesian
                self.expMeth = Bayesian(latent_dim=self.latent_dim)

            else:
                from ExplorationMethods.Slerp import Slerp
                self.expMeth = Slerp(latent_dim=self.latent_dim)

        except ImportError as e:
            raise ImportError(f"Exploration method import failed: {e}")


    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Run Flux2 Klein Inference with HiL Exploration"
        )

        parser.add_argument(
            "--prompt",
            type=str,
            default="a little girl wearing a bright yellow dress and a copper crown is riding a badger through a field of flowers",
            help="生成する画像のプロンプト"
        )

        parser.add_argument(
            "--ExplorationMethod",
            type=int,
            choices=[1, 2, 3],
            default=1,
            help="潜在空間の探索手法; 1=CMAES, 2=Bayesian, 3=Slerp"
        )

        parser.add_argument(
            "--params",
            type=str,
            choices=["4B", "9B"],
            default="4B",
            help="生成モデルのパラメータ数"
        )

        parser.add_argument(
            "--batch",
            type=int,
            default=8,
            help="生成する候補画像数"
        )

        parser.add_argument(
            "--subject",
            type=str,
            default="testSubject",
            help="被験者名"
        )

        return parser.parse_args()

    def generate_flux_image(self, prompt, seed=None, new_vec=False):
        """
        Generates an image using either a text prompt or a pre-saved latent conditioning.
        
        Args:
            prompt (str): The text prompt (used only if latent=False).
            seed (int): Random noise seed.
            latent (bool): If True, uses the LoadConditioning workflow. 
                           If False (default), uses the standard Prompt workflow.
        """
        # 1. Select the appropriate template
        # You should load 'flux2_latent_recursive.json' into self.latent_workflow_template in __init__
        if new_vec:
            workflow = self.latent_workflow_template.copy()
            # Node 83 is the LoadConditioning node in your new JSON
            # We set the path to where your CMA-ES saves the .cond file
            cond_filename = "ComfyUI_latent.cond"
            workflow["83"]["inputs"]["conditioning_path"] = f"conditioning/{cond_filename}"
        else:
            workflow = self.workflow_template.copy()
            workflow[self.prompt_node]["inputs"]["value"] = prompt

        # 2. Handle the Seed (Common to both workflows)
        if seed is None:
            seed = self.rng.randint(1, 10**15)
        
        # Node "75:73" remains the RandomNoise node in both of your JSONs
        workflow["75:73"]["inputs"]["noise_seed"] = seed

        # 3. Construct the Payload
        payload = {
            "prompt": workflow,
            "client_id": str(uuid.uuid4())
        }

        # 4. API Call
        try:
            response = self.session.post(self.url, json=payload)
            response.raise_for_status()
            return response.json()["prompt_id"]

        except requests.exceptions.ConnectionError:
            print("[Network Error] ComfyUI connection failed. Check if server is running.")
        except Exception as e:
            print(f"[Generation error] {e}")

        return None
    
    def get_latent(self):
        csv_file = os.path.join(self.csv_path, "latent.csv")
        
        if not os.path.exists(csv_file):
            print(f"latent.csv not found at : {csv_file}")

        try:
            data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
            
            # Ensure data is treated as 2D even if there's only one row
            if data.ndim == 1:
                data = data.reshape(1, -1)
                
            last_latent = data[-1]
            
            # Print length so you can set your latent_dim in the next run
            print(f"Vector Length: {len(last_latent)}")
            
            if self.expMeth.mean is None:
                self.expMeth.set_initial_mean(last_latent)
                print(">>> CMA-ES Mean initialized")
            
            return last_latent
        except Exception as e:
            print(f"[Read Error] Could not read {csv_file}: {e}")
            return None
    
    def get_mock_eeg_scores(self):
        """Simulates an ERP 'Winner' selection (User picks index 3 out of 8)."""
        # In research, replace this with your EEG trigger/classifier output
        scores = [0.0] * self.args.batch
        winner_idx = 2 # Image 3
        scores[winner_idx] = 1.0
        return scores
    

if __name__ == "__main__":

    flux = FluxInference()
    new_vec = None

    print("Press ENTER to generate an image.")
    print("Press ESC to exit.")

    while True:

        if keyboard.is_pressed("esc"):
            print("Exiting...")
            break

        if keyboard.is_pressed("enter"):
            # If new_vec exists, we tell flux to use the latent-recursive JSON
            is_latent_mode = True if new_vec is not None else False
            
            for _ in range(flux.args.batch):
                # Corrected parameter name 'new_vec' instead of 'latent' to match your function sig
                pid = flux.generate_flux_image(flux.args.prompt, new_vec=is_latent_mode)
                
                if pid:
                    print(f"Queued prompt: {pid}")

            print("\n>>> Wait for generation. Press SPACE to select winner.")
            keyboard.wait("space")
            
            # 1. Get EEG Result (Scores)
            scores = flux.get_mock_eeg_scores()
            
            # 2. Sync history to get current state (this initializes expMeth.mean)
            last_vec_from_csv = flux.get_latent() 
            
            # 3. Trigger the update logic
            # Passing scores updates the manifold and returns the updated mean vector
            new_vec = flux.expMeth(scores)
            
            print(f">>> Manifold updated. Next batch will use evolved latent.")


        keyboard.wait("enter", suppress=False)