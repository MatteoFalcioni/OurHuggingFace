import os
import io
import torch
import base64
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import numpy as np

# https://huggingface.co/ByteDance/Sa2VA-8B#quick-start

# --- CONFIGURATION ---
# Point exactly to your folder
MODEL_PATH = "../models/Sa2VA-8B" 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}\n")

app = FastAPI()

# --- 1. LOAD MODEL (Global Scope) ---
print(f"Loading model from {MODEL_PATH}...")

# trust_remote_code=True is REQUIRED because the model code 
# (modeling_sa2va_chat.py) is inside that folder.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
).to(DEVICE)

model.eval()
print("Model loaded and ready!")

# --- HELPER: Numpy Mask -> Base64 ---
def numpy_mask_to_base64(mask_array):
    # mask_array shape is likely (1, H, W)
    # 1. Squeeze to remove the channel dim -> (H, W)
    mask_sq = np.squeeze(mask_array)
    
    # 2. Normalize to 0-255 uint8 for image saving
    # If mask is 0/1 boolean, * 255 makes it black/white
    mask_uint8 = (mask_sq * 255).astype(np.uint8)
    
    # 3. Create PIL Image
    pil_image = Image.fromarray(mask_uint8)
    
    # 4. Convert to Base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- 2. API ENDPOINT ---
@app.post("/predict")
async def predict(
    image: UploadFile = File(...), 
    prompt: str = Form(...)
):
    try:
        # Read Image from Network
        contents = await image.read()
        raw_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # prepend the <image> suffyx to the prompt
        prompt = '<image>' + prompt

        # --- INFERENCE LOGIC ---
        input_dict = {
            'image' : raw_image,
            'text' : prompt,
            'tokenizer' : tokenizer 
        }
        
        with torch.no_grad():
            return_dict = model.predict_forward(**input_dict)
        
        answer = return_dict['prediction']  # the text format answer
        masks_list = return_dict['prediction_masks'] # segmentation masks, list(np.array(1, h, w), ...)

        # --- PROCESS RESULTS ---
        # Convert the list of numpy masks to a list of base64 strings
        encoded_masks = []
        if masks_list is not None:
            for mask in masks_list:
                encoded_masks.append(numpy_mask_to_base64(mask))

        return {
            "status": "success",
            "text": answer,
            "masks_base64": encoded_masks # Returning a LIST now
        }

    except Exception as e:
        import traceback
        traceback.print_exc() # Print error to remote terminal for debugging
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Host 0.0.0.0 is required to listen on all interfaces (including the tunnel)
    uvicorn.run(app, host="0.0.0.0", port=8000)
