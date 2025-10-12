# AUTHOR
ELCHIN HUSEYNOV

# 🍌 Nano Banana -- AI Image Variation & GIF Generator

**Nano Banana** is a command-line tool that generates **AI-powered
variations of a person's photo** using **Google Gemini**, then stitches
them into an animated **GIF**.

Perfect for experimenting with **expressions, styles, and creative
variations** while keeping the person's size, orientation, and aspect
ratio consistent.

------------------------------------------------------------------------

## ✨ Features

-   🎭 Generate multiple **variations** of a base image with custom
    prompts.\
-   ⚡ Parallel generation for speed (up to 5 at once).\
-   🖼️ Automatically resizes all outputs to match the original.\
-   🎞️ Build an **optimized looping GIF** from generated images.\
-   📂 Supports input from:
    -   Text file (`.txt`, one variation per line)\
    -   JSON file (list or `{ "variations": [...] }`)\
    -   Comma-separated string

------------------------------------------------------------------------

## 📦 Installation

1.  Clone this repository or copy `nano-banana.py`.\

2.  Install dependencies:

    ``` bash
    uv pip install google-genai pillow python-dotenv
    ```

3.  Set your Gemini API key (replace with your actual key):

    ``` bash
    export GEMINI_API_KEY="your-api-key"
    ```

    Or create a `.env` file in the same directory:

        GEMINI_API_KEY=your-api-key

------------------------------------------------------------------------

## 🚀 Usage

### Basic

``` bash
uv run nano-banana.py photo.jpg "happy, sad, surprised"
```

### With variations in a text file

``` bash
uv run nano-banana.py photo.jpg variations.txt
```

### With JSON file

``` bash
uv run nano-banana.py photo.jpg variations.json custom_output/
```

### GIF-only mode

Regenerate a GIF from already existing images in a folder:

``` bash
uv run nano-banana.py photo.jpg variations.txt --gif-only
```

------------------------------------------------------------------------

## ⚙️ Options

  -------------------------------------------------------------------------------
  Option                                      Description
  ------------------------------------------- -----------------------------------
  `input_image`                               Path to base image of a person

  `variations`                                File path (`.txt`/`.json`) or
                                              comma-separated list

  `output_dir`                                Directory to save results (default:
                                              `output`)

  `--duration`                                Frame duration in GIF (ms, default:
                                              750)

  `--model`                                   Gemini model to use (default:
                                              `gemini-2.5-flash-image-preview`)

  `--gif-quality`                             Color palette size for GIF (64,
                                              128, 256; default: 256)

  `--gif-only`                                Skip generation, just rebuild GIF
                                              from existing images
  -------------------------------------------------------------------------------

------------------------------------------------------------------------

## 📂 Output

-   Original image is copied as `original.jpg/png` in `output/`.\
-   Variations saved as `variation_000_0.png`, `variation_001_0.png`,
    etc.\
-   Final GIF saved as `variations.gif`.

Example structure:

    output/
    ├── original.jpg
    ├── variation_000_0.png
    ├── variation_001_0.png
    ├── variation_002_0.png
    └── variations.gif

------------------------------------------------------------------------

## 📝 Notes

-   Requires **Gemini API access with image generation enabled**.\
-   Default model: `gemini-2.5-flash-image-preview` (update if
    deprecated).\
-   Prompts should describe **modifications only** (e.g., "wearing
    sunglasses", "angry expression").

------------------------------------------------------------------------
