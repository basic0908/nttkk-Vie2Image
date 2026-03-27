import torch
import io
import base64
import uvicorn
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from diffusers import Flux2KleinPipeline
from PIL import Image

app = FastAPI()

# ==========================================
# MODEL LOADING SEQUENCE
# ==========================================
print("\n" + "="*50)
print("🚀 INITIALIZING AWS FLUX.2-KLEIN SERVER (High-Speed Mode)")
print("="*50)

HF_TOKEN = "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" # Revoke when done testing!

print("[1/2] Loading FLUX.2-klein-4B model into VRAM...")
start_time = time.time()

# We load the model directly onto the GPU ("cuda") to pin it there permanently.
# This eliminates PCIe transfer delays during generation!
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", 
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
).to("cuda")

print(f"      Model loaded in {time.time() - start_time:.1f} seconds.")

# REMOVED pipe.enable_model_cpu_offload()! 
# The L40S has 48GB VRAM. We want to use it!
print("[2/2] Model pinned to GPU VRAM for maximum speed.")

print("\n✅ KLEIN MODEL LOADED SUCCESSFULLY.")
print("🌐 Starting API Server. Ready to receive prompts!\n")

# ==========================================
# PROGRESS CALLBACK
# ==========================================
# This function runs automatically inside the GPU loop after every step
def step_tracker(pipe, step_index, timestep, callback_kwargs):
    # step_index starts at 0, so we add 1 for display
    print(f"    -> [GPU] Denoising Step {step_index + 1} / 4 complete...")
    return callback_kwargs

# ==========================================
# API ENDPOINTS
# ==========================================
class PromptBatchRequest(BaseModel):
    prompts: List[str]
    seed: int = 42

@app.post("/generate_batch")
async def generate_batch(req: PromptBatchRequest):
    prompts = req.prompts
    batch_size = len(prompts)
    
    generators = [torch.Generator(device="cuda").manual_seed(req.seed + i) for i in range(batch_size)]
    
    print(f"\n[+] Received request: Generating {batch_size} images...")
    gen_start = time.time()
    
    with torch.inference_mode():
        images = pipe(
            prompt=prompts,
            height=1024,               
            width=1024,
            num_inference_steps=4,     
            guidance_scale=1.0,        
            generator=generators,
            callback_on_step_end=step_tracker  # Inject the visual progress tracker
        ).images

    print(f"[-] Generation finished in {time.time() - gen_start:.1f} seconds.")

    print("[+] Encoding images to Base64 (This might take ~1 second)...")
    encoded_images = []
    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        encoded_images.append(img_str)
        
    print(f"[-] Transfer ready. Sending {len(encoded_images)} images back to client.")
    return {"images": encoded_images, "prompts": prompts}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)