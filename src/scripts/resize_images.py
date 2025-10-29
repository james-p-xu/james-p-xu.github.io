#!/usr/bin/env python3
"""
Image resizing script to standardize all images to 1280px maximum width.
Maintains aspect ratios and optimizes file sizes.
"""

import os
import sys
from PIL import Image
from pathlib import Path

def resize_image_to_1280px(input_path, max_width=1280):
    """
    Resize an image to maximum 1280px width while maintaining aspect ratio.
    
    Args:
        input_path: Path to input image
        max_width: Maximum width in pixels (default: 1280)
    
    Returns:
        bool: True if image was resized, False if already small enough
    """
    try:
        with Image.open(input_path) as img:
            original_size = img.size
            original_file_size = os.path.getsize(input_path)
            
            # Calculate new dimensions maintaining aspect ratio
            if original_size[0] > max_width:
                scale_factor = max_width / original_size[0]
                new_width = int(original_size[0] * scale_factor)
                new_height = int(original_size[1] * scale_factor)
                
                # Resize with high-quality resampling
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert RGBA to RGB if no transparency is used (reduces file size)
                if resized_img.mode == 'RGBA':
                    alpha = resized_img.split()[-1]
                    if alpha.getextrema() == (255, 255):  # All pixels are opaque
                        resized_img = resized_img.convert('RGB')
                        print(f"  Converted RGBA to RGB (no transparency)")
                
                # Save with optimization
                resized_img.save(input_path, optimize=True, compress_level=6)
                
                new_file_size = os.path.getsize(input_path)
                reduction_percent = (1 - new_file_size / original_file_size) * 100
                
                print(f"✓ {input_path}")
                print(f"  {original_size[0]}x{original_size[1]} → {new_width}x{new_height}")
                print(f"  {original_file_size:,} → {new_file_size:,} bytes ({reduction_percent:.1f}% reduction)")
                return True
            else:
                print(f"✓ {input_path}: {original_size[0]}x{original_size[1]} (already ≤{max_width}px)")
                return False
                
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False

def resize_gif_to_1280px(input_path, max_width=1280):
    """
    Resize a GIF to maximum 1280px width while maintaining aspect ratio,
    preserving animation.
    """
    try:
        with Image.open(input_path) as img:
            original_size = img.size
            original_file_size = os.path.getsize(input_path)
            
            if original_size[0] <= max_width:
                print(f"✓ {input_path}: {original_size[0]}x{original_size[1]} (already ≤{max_width}px)")
                return False  # No resize needed

            scale_factor = max_width / original_size[0]
            new_width = int(original_size[0] * scale_factor)
            new_height = int(original_size[1] * scale_factor)

            frames = []
            durations = []

            for frame in ImageSequence.Iterator(img):
                # Keep frame mode as is and just resize
                resized_frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                frames.append(resized_frame)
                durations.append(frame.info.get("duration", 100))

            # Save resized GIF
            frames[0].save(
                input_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=img.info.get("loop", 0),
                disposal=2,
                optimize=True
            )

            new_file_size = os.path.getsize(input_path)
            reduction_percent = (1 - new_file_size / original_file_size) * 100
            print(f"✓ {input_path}")
            print(f"  {original_size[0]}x{original_size[1]} → {new_width}x{new_height}")
            print(f"  {original_file_size:,} → {new_file_size:,} bytes ({reduction_percent:.1f}% reduction)")

            return True

    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False

def main():
    """Main function to resize all images in the repository."""
    
    # Define directories to process
    directories = [
        "src/assets/images",
        "public/assets/images",
    ]
    
    # Image extensions to process
    image_extensions = {'.png', '.jpg', '.jpeg'}
    gif_extensions = {'.gif'}
    
    total_processed = 0
    total_resized = 0
    
    print("Resizing all images to 1280px maximum width...")
    print("=" * 60)
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        print(f"\nProcessing directory: {directory}")
        print("-" * 30)
        
        # Find all image files
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                # Skip venv directory
                if 'venv' in str(file_path):
                    continue
                
                file_ext = file_path.suffix.lower()
                
                if file_ext in image_extensions:
                    if resize_image_to_1280px(str(file_path)):
                        total_resized += 1
                    total_processed += 1
                elif file_ext in gif_extensions:
                    if resize_gif_to_1280px(str(file_path)):
                        total_resized += 1
                    total_processed += 1
    
    print("\n" + "=" * 60)
    print("RESIZE SUMMARY")
    print("=" * 60)
    print(f"Images processed: {total_processed}")
    print(f"Images resized: {total_resized}")
    print(f"Images already optimal: {total_processed - total_resized}")
    print("\nAll images now have maximum 1280px width while maintaining aspect ratios!")

if __name__ == "__main__":
    main()
