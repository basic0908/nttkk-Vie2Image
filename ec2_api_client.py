import gradio as gr
import requests
import base64
import io
import json
import time
import os
import csv
import re
from PIL import Image
from openai import OpenAI

# ==========================================
# CONFIGURATION
# ==========================================
AWS_SERVER_URL = "http://{ENTER YOUR EC2 IP}:8000/generate_batch" # <----自分のec2のIPアドレスに書き換える
BATCH_SIZE = 10
LLM_MODEL = "llama3.1"
EXPERIMENT_ROOT_DIR = "./experiments" 

MAX_ITERATIONS = 5

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" 
)

# ==========================================
# HELPER FUNCTIONS (Logging & Saving)
# ==========================================
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def save_iteration_data_with_ratings(base_dir, iter_num, images, prompts, ratings):
    """Used strictly for the 100 Baseline Images (Iter 0) where we auto-fill ratings with 0"""
    iter_dir = os.path.join(base_dir, f"iter_{iter_num}")
    os.makedirs(iter_dir, exist_ok=True)
    
    for idx, img in enumerate(images):
        img_path = os.path.join(iter_dir, f"{idx}.jpg")
        img.save(img_path, format="JPEG", quality=95)
        
    csv_path = os.path.join(base_dir, "history.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iter_num", "id", "prompt", "rating"])
        for idx, (p, r) in enumerate(zip(prompts, ratings)):
            writer.writerow([iter_num, idx, p, r])

def save_iteration_data_unrated(base_dir, iter_num, images, prompts):
    """Used for BCI Loop. Saves images and writes CSV with EMPTY ratings for MATLAB to fill."""
    iter_dir = os.path.join(base_dir, f"iter_{iter_num}")
    os.makedirs(iter_dir, exist_ok=True)
    
    for idx, img in enumerate(images):
        img_path = os.path.join(iter_dir, f"{idx}.jpg")
        img.save(img_path, format="JPEG", quality=95)
        
    csv_path = os.path.join(base_dir, "history.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iter_num", "id", "prompt", "rating"])
        for idx, p in enumerate(prompts):
            writer.writerow([iter_num, idx, p, ""]) # Empty rating!
            
    print(f"[*] Saved Iteration {iter_num} data. Awaiting MATLAB EEG ratings in CSV...")

def wait_for_matlab_ratings(csv_path, iter_num, batch_size):
    """Polls the CSV file every 2 seconds until MATLAB fills in the ratings for the current iteration."""
    print(f"[*] Polling {csv_path} for Iteration {iter_num} ratings...")
    
    while True:
        time.sleep(2) # Prevent CPU thrashing while waiting
        try:
            with open(csv_path, mode='r', encoding='utf-8-sig') as f:
                lines = list(csv.reader(f))
            
            # Extract only the rows that belong to the current iteration
            current_iter_rows = [row for row in lines[1:] if row[0] == str(iter_num)]
            
            # If the CSV has our 10 rows...
            if len(current_iter_rows) == batch_size:
                all_rated = True
                ratings = []
                
                for row in current_iter_rows:
                    if len(row) < 4: # Safety check
                        all_rated = False
                        break
                        
                    rating_str = row[3].strip()
                    if not rating_str: # If the cell is still empty
                        all_rated = False
                        break
                        
                    try:
                        ratings.append(float(rating_str))
                    except ValueError:
                        all_rated = False
                        break
                
                if all_rated:
                    print(f"[*] Success! MATLAB EEG ratings received: {ratings}")
                    return ratings
                    
        except Exception as e:
            pass

# ==========================================
# API CALLS
# ==========================================
def call_aws_generator(prompts, progress=gr.Progress()):
    """Sends prompts to AWS. If it fails, it waits and retries infinitely until successful."""
    print(f"[*] Sending {len(prompts)} prompts to AWS...")
    attempt = 1
    
    while True:
        try:
            progress(0.4, desc=f"Generating {BATCH_SIZE} images on AWS (Attempt {attempt})...")
            start_time = time.time()
            
            response = requests.post(AWS_SERVER_URL, json={"prompts": prompts, "seed": 42}, timeout=300)
            response.raise_for_status() 
            data = response.json()
            
            progress(0.9, desc="Decoding received images...")
            images = []
            for b64_str in data["images"]:
                img_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
                
            print(f"[*] Successfully received {len(images)} images in {time.time() - start_time:.1f} seconds.")
            return images
            
        except requests.exceptions.RequestException as e:
            print(f"[!] ERROR connecting to AWS Server on attempt {attempt}: {e}")
            print("[*] Retrying in 5 seconds...")
            progress(0.4, desc=f"AWS connection failed. Retrying in 5s (Attempt {attempt})...")
            time.sleep(5)
            attempt += 1
            
        except Exception as e:
            print(f"[!] FATAL ERROR processing AWS response: {e}")
            print("[*] Retrying in 5 seconds...")
            progress(0.4, desc=f"Data processing failed. Retrying in 5s (Attempt {attempt})...")
            time.sleep(5)
            attempt += 1

def generate_appo_prompts(base_concept, history=None, hint=None, progress=gr.Progress()):
    progress(0.1, desc=f"Thinking of {BATCH_SIZE} prompt variations (Local LLM)...")
    
    if not history:
        sys_msg = f"You are an expert prompt engineer. The user wants to generate an image based on a concept. Create exactly {BATCH_SIZE} highly diverse, detailed text-to-image prompts exploring different visual styles, lighting, and compositions for this concept. Output ONLY a valid JSON array of {BATCH_SIZE} strings. Do not include markdown formatting, explanations, or any other text."
        user_msg = f"Concept: {base_concept}"
        if hint:
            user_msg += f"\nNote: {hint}"
            
    else:
        filtered_history = []
        
        iterations = sorted(list(set(item['iter'] for item in history)))
        
        for iter_num in iterations:
            iter_items = [item for item in history if item['iter'] == iter_num]
            
            # Sort them by rating descending, and take the top 3
            top_3_iter = sorted(iter_items, key=lambda x: x['rating'], reverse=True)[:3]
            
            # Append these 3 winners to our final filtered list
            filtered_history.extend(top_3_iter)
        
        print(f"[*] Filtered history: Accumulated {len(filtered_history)} top-rated prompts across {len(iterations)} iterations.")

        history_text = f"=== TOP 3 PROMPTS FROM EACH PAST ITERATION ===\n"
        for item in filtered_history:
            history_text += f"[Iter {item['iter']}] Rating: {item['rating']}/100 | Prompt: {item['prompt']}\n"
            
        sys_msg = f"""You are an expert prompt optimizer following an evolutionary APPO framework. 
        You are given a 'Base Concept' and a curated history showing the TOP 3 highest-rated prompts from EACH previous iteration.
        Each prompt has an objective, EEG-derived rating from 1 to 100.
        
        CRITICAL INSTRUCTION: Treat the rating as a WEIGHT. HIGHER ratings mean the user liked it MORE.
        
        Generate exactly {BATCH_SIZE} new prompts. Follow these strategies:
        1. ANALYZE EVOLUTION: Look at how the highly-rated prompts have evolved across iterations. Heavily favor, retain, and blend core visual elements, styles, and themes from the best performing prompts.
        2. EXPAND: Introduce slight variations to the highly-rated concepts to further explore the user's ideal latent space.
        
        Output ONLY a valid JSON array of {BATCH_SIZE} strings. Do not include markdown formatting, explanations, or any other text."""
        
        user_msg = f"Base Concept: {base_concept}\n\n{history_text}"

    print(f"[*] Asking local LLM for {BATCH_SIZE} prompt variations...")
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.8,
        max_tokens=2048
    )
    
    try:
        content = response.choices[0].message.content
        start = content.find('[')
        end = content.rfind(']') + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in response")
            
        prompts = json.loads(content[start:end])
        
        while len(prompts) < BATCH_SIZE:
            prompts.append(prompts[0])
            
        return prompts[:BATCH_SIZE]
        
    except Exception as e:
        print(f"[!] LLM Parsing Error: {e}")
        return [f"{base_concept}, detailed aesthetic variation {i}" for i in range(BATCH_SIZE)]

# ==========================================
# GRADIO UI LOGIC
# ==========================================
def generate_100_images(base_concept, subject_name, progress=gr.Progress()):
    if not base_concept.strip():
        raise gr.Error("Please enter a concept first.")
    if not subject_name.strip():
        raise gr.Error("Please enter a subject name first.")
        
    print(f"\n=== GENERATING 100 IMAGES FOR DECODER (Iter 0) ===")
    
    os.makedirs(EXPERIMENT_ROOT_DIR, exist_ok=True)
    safe_subject = sanitize_filename(subject_name.strip())
    safe_concept = sanitize_filename(base_concept.strip())
    base_dir = os.path.join(EXPERIMENT_ROOT_DIR, f"{safe_subject}_{safe_concept[:30]}")
    os.makedirs(base_dir, exist_ok=True)
    
    all_prompts = []
    all_images = []
    all_ratings = []
    
    for i in range(10):
        progress(i / 10.0, desc=f"Generating batch {i+1}/10 (100 images total)...")
        hint = f"Make this set #{i+1} highly distinct and diverse from previous sets to explore completely new aesthetic spaces."
        
        prompts = generate_appo_prompts(base_concept, hint=hint, progress=progress)
        images = call_aws_generator(prompts, progress=progress)
        
        all_prompts.extend(prompts)
        all_images.extend(images)
        all_ratings.extend([0] * BATCH_SIZE) 
        
    save_iteration_data_with_ratings(base_dir, 0, all_images, all_prompts, all_ratings)
    
    return all_images, all_prompts, [], f"**Status:** Baseline 100 images generated (Iter 0). Saved to {base_dir}", base_dir

def run_bci_auto_loop(base_concept, subject_name, base_dir, progress=gr.Progress()):
    """This generator function acts as the infinite BCI loop"""
    if not base_concept.strip() or not subject_name.strip():
        raise gr.Error("Missing concept or subject name.")
        
    safe_subject = sanitize_filename(subject_name.strip())
    safe_concept = sanitize_filename(base_concept.strip())
    expected_dir = os.path.join(EXPERIMENT_ROOT_DIR, f"{safe_subject}_{safe_concept[:30]}")
    
    if not base_dir:
        base_dir = expected_dir
        
    if not os.path.exists(base_dir):
        raise gr.Error(f"Error: Directory '{base_dir}' not found! Generate 100 baseline images first or ensure the folder name matches your inputs.")
        
    print(f"\n=== STARTING AUTOMATED BCI LOOP (Max {MAX_ITERATIONS} iterations) ===")
    
    csv_path = os.path.join(base_dir, "history.csv")
    history_state = []
    
    for iter_num in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- ITERATION {iter_num} ---")
        
        prompts = generate_appo_prompts(base_concept, history=history_state if history_state else None, progress=progress)
        images = call_aws_generator(prompts, progress=progress)
        
        save_iteration_data_unrated(base_dir, iter_num, images, prompts)
        
        status_msg = f"**Iter {iter_num} Complete.** Waiting for MATLAB to write ratings to `history.csv`..."
        yield images, prompts, history_state, status_msg, base_dir
        
        ratings = wait_for_matlab_ratings(csv_path, iter_num, BATCH_SIZE)
        
        for p, r in zip(prompts, ratings):
            history_state.append({'iter': iter_num, 'prompt': p, 'rating': r})
            
        status_msg = f"**Iter {iter_num} Ratings Received!** Optimizing Iter {iter_num + 1} via LLM..."
        yield images, prompts, history_state, status_msg, base_dir

    yield images, prompts, history_state, f"**Experiment Complete.** Reached max {MAX_ITERATIONS} iterations.", base_dir

# ==========================================
# GRADIO LAYOUT
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# APPO-BCI: Passive Preference Prompt Optimization")
    gr.Markdown("Based on CHI 2026. This version actively monitors `history.csv` for external EEG ratings from MATLAB and dynamically feeds the **Top 3 prompts from EVERY past iteration** back into the LLM.")
    
    with gr.Row():
        base_concept_input = gr.Textbox(label="Initial Concept", scale=3)
        subject_name_input = gr.Textbox(label="Subject Name", scale=2)
        with gr.Column(scale=1):
            generate_100_btn = gr.Button("1. Generate 100 Images (Baseline)", variant="secondary")
            start_bci_btn = gr.Button("2. Start BCI Auto-Loop", variant="primary")
    
    current_prompts_state = gr.State([])
    full_history_state = gr.State([]) 
    base_dir_state = gr.State("")
    
    with gr.Row():
        with gr.Column(scale=3):
            gallery = gr.Gallery(
                label="Generated Candidates", 
                show_label=True, 
                elem_id="gallery", 
                columns=2,          
                rows=5,            
                height="80vh",      
                object_fit="contain",
                allow_preview=False 
            )
            
            with gr.Row():
                status_text = gr.Markdown("**Status:** Awaiting start...")

        with gr.Column(scale=1):
            prompt_debugger = gr.JSON(label="Current Iteration Prompts")

    generate_100_btn.click(
        fn=generate_100_images,
        inputs=[base_concept_input, subject_name_input],
        outputs=[gallery, prompt_debugger, full_history_state, status_text, base_dir_state]
    )

    start_bci_btn.click(
        fn=run_bci_auto_loop,
        inputs=[base_concept_input, subject_name_input, base_dir_state],
        outputs=[gallery, prompt_debugger, full_history_state, status_text, base_dir_state]
    )

if __name__ == "__main__":
    print(f"Starting APPO-BCI UI (Top 3 Per-Iteration Filtering, Max Iterations: {MAX_ITERATIONS})...")
    demo.launch(server_name="127.0.0.1", server_port=7860)