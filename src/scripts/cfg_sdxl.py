import torch
from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np

def generate_cfg_grids():
    # Load SDXL pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Detailed prompt for CFG evaluation
    prompt = "a majestic golden retriever sitting in a sunny meadow with wildflowers, professional pet photography, shallow depth of field, warm natural lighting, highly detailed fur texture, photorealistic"
    negative_prompt = "blurry, low quality, distorted, deformed, cartoon, anime, painting, sketch, ugly, duplicate, extra limbs"
    
    # CFG values to test - demonstrating the classic CFG trade-off
    cfg_values = [0.0, 3.0, 10.0]
    cfg_names = ["cfg_0", "cfg_3", "cfg_10"]
    
    # Use 9 different seeds for each 3x3 grid to show diversity/consistency
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324]
    
    print("Generating three 3x3 grids for CFG comparison...")
    
    for cfg_idx, (cfg, cfg_name) in enumerate(zip(cfg_values, cfg_names)):
        print(f"\nGenerating 3x3 grid for CFG={cfg}...")
        
        images = []
        
        # Generate 9 images with same CFG, different seeds
        for i, seed in enumerate(seeds):
            print(f"  Generating image {i+1}/9 (seed={seed})...")
            
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=cfg,
                num_inference_steps=30,
                width=512,
                height=512,
                generator=torch.Generator(device="cuda").manual_seed(seed)
            ).images[0]
            
            images.append(image)
        
        # Create 3x3 grid
        img_size = 512
        gap = 10
        
        grid_width = 3 * img_size + 2 * gap
        grid_height = 3 * img_size + 2 * gap
        
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Place images in 3x3 grid
        for i, img in enumerate(images):
            row = i // 3
            col = i % 3
            
            x = col * (img_size + gap)
            y = row * (img_size + gap)
            
            grid_image.paste(img, (x, y))
        
        # Save individual grid
        output_path = f"dog_grid_{cfg_name}.png"
        grid_image.save(output_path)
        print(f"  Grid saved to: {output_path}")
    
    print(f"\nAll CFG comparison grids completed!")
    print(f"Files saved: dog_grid_cfg_0.png, dog_grid_cfg_3.png, dog_grid_cfg_10.png")
    
    return f"Generated 3 grids for CFG values: {cfg_values}"

if __name__ == "__main__":
    print("Starting SDXL CFG comparison generation...")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    result = generate_cfg_grids()
    print("CFG comparison generation completed!")
    print(result)
