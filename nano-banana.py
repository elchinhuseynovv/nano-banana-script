import base64
import mimetypes
import os
import sys
import argparse
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from PIL import Image
import json
import dotenv
import glob

dotenv.load_dotenv()


def save_binary_file(file_name: str, data: bytes) -> None:
    """Save binary data to file."""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")


def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Load image and return as base64 string with mime type."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image/'):
        raise ValueError(f"File is not a valid image: {image_path}")
    
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return base64_data, mime_type


def load_variations(variations_input: str) -> List[str]:
    """Load variations from file or parse from string."""
    variations = []
    
    # Check if it's a file path
    if os.path.exists(variations_input):
        with open(variations_input, 'r') as f:
            content = f.read().strip()
            
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    variations = data
                elif isinstance(data, dict) and 'variations' in data:
                    variations = data['variations']
                else:
                    raise ValueError("JSON must be a list or contain 'variations' key")
            except json.JSONDecodeError:
                # Parse as plain text (one variation per line)
                variations = [line.strip() for line in content.split('\n') if line.strip()]
    else:
        # Treat as comma-separated string
        variations = [v.strip() for v in variations_input.split(',') if v.strip()]
    
    if not variations:
        raise ValueError("No variations provided")
    
    return variations


def generate_variation(client, model: str, base_image_b64: str, base_mime_type: str, 
                      variation_prompt: str, output_dir: Path, variation_index: int) -> str:
    """Generate a single variation of the image."""
    
    # Create the prompt that includes the base image and variation request
    prompt = f"""I have provided an image of a person. Please create a variation of this person with the following modification: {variation_prompt}
Keep the position and orientation of the person the same as the original image. The image should have the same aspect ratio and the person should be the same size as the original image. This is extremely important. THE PERSON SHOULD BE THE SAME SIZE AS THE ORIGINAL IMAGE AND THE IMAGE SHOULD HAVE THE SAME ASPECT RATIO AS THE ORIGINAL IMAGE."""
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(
                    data=base64.b64decode(base_image_b64),
                    mime_type=base_mime_type
                )
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )

    print(f"Generating variation {variation_index + 1}: {variation_prompt}")
    
    file_index = 0
    output_file = None
    
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
                
            if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
                file_name = f"variation_{variation_index:03d}_{file_index}"
                file_index += 1
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                data_buffer = inline_data.data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                
                output_file = output_dir / f"{file_name}{file_extension}"
                save_binary_file(str(output_file), data_buffer)
            else:
                if chunk.text:
                    print(f"Response: {chunk.text}")
                    
    except Exception as e:
        print(f"Error generating variation {variation_index + 1}: {e}")
        return None
        
    return str(output_file) if output_file else None


def collect_existing_images(output_dir: Path) -> List[str]:
    """Collect all existing images in the output directory."""
    image_paths = []
    
    # Add original image if it exists
    for ext in ['.jpeg', '.jpg', '.png']:
        original_path = output_dir / f"original{ext}"
        if original_path.exists():
            image_paths.append(str(original_path))
            break
    
    # Add variation images, sorted by number
    variation_pattern = str(output_dir / "variation_*_0.png")
    variation_files = sorted(glob.glob(variation_pattern))
    image_paths.extend(variation_files)
    
    return image_paths


def create_gif_from_images(image_paths: List[str], output_path: str, duration: int = 1000, quality: int = 256) -> None:
    """Create an optimized GIF from a list of image paths with consistent sizing."""
    if not image_paths:
        print("No images to create GIF from")
        return
    
    # Filter out None values and non-existent files
    valid_paths = [path for path in image_paths if path and os.path.exists(path)]
    
    if not valid_paths:
        print("No valid images found for GIF creation")
        return
    
    try:
        print(f"Processing {len(valid_paths)} images for GIF...")
        
        # Load first image to get target dimensions
        first_img = Image.open(valid_paths[0])
        target_size = first_img.size
        print(f"Target size: {target_size[0]}x{target_size[1]}")
        
        images = []
        for i, path in enumerate(valid_paths):
            img = Image.open(path)
            
            # Resize to match first image dimensions if different
            if img.size != target_size:
                print(f"Resizing image {i+1} from {img.size} to {target_size}")
                # Use LANCZOS for high quality resizing
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary (for GIF compatibility)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Optimize for GIF: reduce colors using adaptive palette
            img = img.quantize(colors=quality, method=Image.Quantize.MEDIANCUT)
            images.append(img)
        
        if images:
            print("Creating optimized GIF...")
            # Save as GIF with optimization
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=True,  # Enable optimization for smaller file size
                disposal=2      # Clear frame before next for better compression
            )
            print(f"GIF created: {output_path}")
        else:
            print("No valid images to create GIF")
            
    except Exception as e:
        print(f"Error creating GIF: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate variations of a person image and create a GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python nano-banana-mix.py photo.jpg variations.txt
    python nano-banana-mix.py photo.jpg "happy, sad, surprised, angry"
    python nano-banana-mix.py photo.jpg variations.json custom_output/
        """
    )
    
    parser.add_argument("input_image", nargs='?', help="Path to input image of person")
    parser.add_argument("variations", nargs='?', help="File path or comma-separated list of variations")
    parser.add_argument("output_dir", nargs='?', default="output", help="Output directory (default: output)")
    parser.add_argument("--duration", type=int, default=750, help="GIF frame duration in ms (default: 750)")
    parser.add_argument("--model", default="gemini-2.5-flash-image-preview", help="Gemini model to use")
    parser.add_argument("--gif-quality", type=int, default=256, choices=[64, 128, 256], help="GIF color palette size for quality vs speed (default: 256)")
    parser.add_argument("--gif-only", action="store_true", help="Only regenerate GIF from existing images in output directory")
    
    args = parser.parse_args()
    
    # Handle GIF-only mode
    if args.gif_only:
        if not args.output_dir or args.output_dir == "output":
            args.output_dir = input("Enter directory with existing images: ").strip() or "output"
        
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"Error: Directory {output_dir} does not exist")
            sys.exit(1)
        
        # Collect existing images
        existing_images = collect_existing_images(output_dir)
        if not existing_images:
            print(f"No images found in {output_dir}")
            sys.exit(1)
        
        print(f"Found {len(existing_images)} existing images")
        
        # Create GIF from existing images
        gif_path = output_dir / "variations.gif"
        print(f"Creating GIF with {len(existing_images)} images at {args.duration}ms per frame...")
        create_gif_from_images(existing_images, str(gif_path), args.duration, args.gif_quality)
        
        print(f"\n=== GIF Regenerated ===")
        print(f"Images used: {len(existing_images)}")
        print(f"Frame duration: {args.duration}ms")
        print(f"GIF created: {gif_path}")
        return
    
    # Interactive mode if no arguments provided
    if not args.input_image:
        print("=== Nano Banana Mix - Interactive Mode ===")
        args.input_image = input("Enter path to input image: ").strip()
        args.variations = input("Enter variations file path or comma-separated prompts: ").strip()
        args.output_dir = input("Enter output directory (default: output): ").strip() or "output"
    
    # Validate API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is required")
        print("Set it with: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    try:
        # Load base image
        print(f"Loading base image: {args.input_image}")
        base_image_b64, base_mime_type = load_image_as_base64(args.input_image)
        
        # Load variations
        print(f"Loading variations from: {args.variations}")
        variations = load_variations(args.variations)
        print(f"Found {len(variations)} variations: {variations}")
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Copy original image to output directory
        original_output = output_dir / f"original{Path(args.input_image).suffix}"
        with open(args.input_image, 'rb') as src, open(original_output, 'wb') as dst:
            dst.write(src.read())
        print(f"Original image copied to: {original_output}")
        
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
        # Generate variations in parallel
        generated_images = [str(original_output)]  # Start with original
        
        print(f"Generating {len(variations)} variations in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(variations), 5)) as executor:
            # Submit all tasks
            future_to_variation = {
                executor.submit(
                    generate_variation,
                    client, args.model, base_image_b64, base_mime_type,
                    variation, output_dir, i
                ): (i, variation) for i, variation in enumerate(variations)
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_variation):
                completed_count += 1
                i, variation = future_to_variation[future]
                try:
                    result = future.result()
                    if result:
                        generated_images.append(result)
                    print(f"Progress: {completed_count}/{len(variations)} variations completed")
                except Exception as e:
                    print(f"Error generating variation {i + 1} ('{variation}'): {e}")
        
        # Create GIF
        gif_path = output_dir / "variations.gif"
        print(f"Creating GIF with {len(generated_images)} images...")
        create_gif_from_images(generated_images, str(gif_path), args.duration, args.gif_quality)
        
        print(f"\n=== Results ===")
        print(f"Generated images: {len(generated_images)}")
        print(f"Output directory: {output_dir}")
        print(f"GIF created: {gif_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()