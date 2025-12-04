import os
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer

# --- CONFIG ---
IMAGE_PATH = "./test_images/img1.jpeg" 
OUTPUT_DIR = "./output_results"          # New: Where to save results
MODEL_PATH = "../models/Sa2VA-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "segment the windows of the building you see in the image. Only the windows." 

def test_inference():
    print(f"--- Starting Local Test ---")
    print(f"Device: {DEVICE}")
    
    # 0. Setup Output Dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Check Image
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        return
    
    try:
        raw_image = Image.open(IMAGE_PATH).convert("RGB")
        print(f"Image loaded: {raw_image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            use_flash_attn=True
        ).to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"CRASH during model load: {e}")
        return

    # 3. Run Inference
    full_prompt = '<image>' + PROMPT
    print(f"Prompt: {full_prompt}")
    
    input_dict = {
        'image': raw_image,
        'text': full_prompt,
        'past_text' : '',
        'tokenizer': tokenizer,
        'mask_prompts': None,
    }

    print("Running predict_forward...")
    try:
        with torch.no_grad():
            return_dict = model.predict_forward(**input_dict)
        
        answer = return_dict['prediction']
        masks = return_dict['prediction_masks']
        
        print("\n--- SUCCESS ---")
        print(f"Text Answer: {answer}")
        
        if masks is not None and len(masks) > 0:
            print(f"Generated {len(masks)} masks. Saving to {OUTPUT_DIR}...")
            
            # Iterate through masks (in case multiple objects were found)
            for i, mask_np in enumerate(masks):
                
                # 1. Check current shape
                # Expected: (1, H, W)
                print(f"  Mask {i} raw shape: {mask_np.shape}")
                
                # 2. Remove the channel dimension (Dim 0)
                # (1, H, W) -> (H, W)
                mask_np = mask_np.squeeze(0)
                
                # 3. Validation
                # If the H or W is 1 (like your previous error showed), this will still process, 
                # but the output image will be a 1-pixel line.
                h, w = mask_np.shape
                print(f"  -> Processing as {h}x{w} image")

                # 4. Scale to 0-255
                mask_img_data = (mask_np * 255).astype(np.uint8)
                
                # 5. Create Image
                mask_img = Image.fromarray(mask_img_data)
                
                # Save Raw Mask
                mask_filename = f"mask_{i}.png"
                mask_path = os.path.join(OUTPUT_DIR, mask_filename)
                mask_img.save(mask_path)
                print(f"  -> Saved raw mask: {mask_path}")

                # Create Overlay (Red Tint) for easier comparison
                # Resize mask to match original image if needed (Sa2VA output matches input usually, but good practice)
                if mask_img.size != raw_image.size:
                     mask_img = mask_img.resize(raw_image.size, resample=Image.NEAREST)

                overlay = raw_image.copy()
                # Create a solid red layer
                red_layer = Image.new("RGB", raw_image.size, (255, 0, 0))
                # Paste red layer onto original using the mask as transparency
                overlay.paste(red_layer, (0,0), mask_img)
                # Blend it so you can see through the red
                final_comp = Image.blend(raw_image, overlay, alpha=0.5)
                
                comp_filename = f"overlay_{i}.jpg"
                comp_path = os.path.join(OUTPUT_DIR, comp_filename)
                final_comp.save(comp_path)
                print(f"  -> Saved overlay: {comp_path}")

        else:
            print("No masks generated.")

    except torch.cuda.OutOfMemoryError:
        print("CRASH: CUDA Out of Memory! The model is too big for this GPU.")
    except Exception as e:
        print(f"CRASH during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()