import base64
import io
import mimetypes
import os
import subprocess
import sys

import openai
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Set OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Settings for processing images
image_folder = "images"
image_longest_side_in_px = 2000  # Max length for the longer side of the image
upscale_small_images = True  # Upscale images smaller than the longest side

# Global token usage summary
token_usage_summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

if not openai_api_key:
    print("No OpenAI API key provided. Please set the OPENAI_API_KEY in the .env file.")
    sys.exit(1)

openai.api_key = openai_api_key


def send_image_to_openai_api(image_base64):
    # instructions = "Generate a descriptive caption for this image."
    instructions = (
        "Generate a concise, descriptive caption for the image, focusing on key elements and features. "
        "Analyze all elements in the image, understanding both the overall composition and individual components. "
        "Describe elements using single words or brief phrases, avoiding long sentences, and separate them with commas. "
        "Include tags for poses, orientations, and background styles where relevant. "
        "Refrain from using specific names of locations, brands, or individuals in your descriptions. Replace these with generic terms that broadly categorize the subject."
        "Example: For an image with an in-vehicle infotainment system, use 'user interface, clean design, navigation, music player, in-vehicle infotainment, white style, button bar on bottom.'"
    )
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64, "detail": "low"},
                    },
                ],
            }
        ],
        "max_tokens": 2000,
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        response_data = response.json()
        description = response.json()["choices"][0]["message"]["content"]

        # Update global token usage summary
        token_usage = response_data.get("usage", {})
        global token_usage_summary
        token_usage_summary["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
        token_usage_summary["completion_tokens"] += token_usage.get(
            "completion_tokens", 0
        )
        token_usage_summary["total_tokens"] += token_usage.get("total_tokens", 0)
        print(f"Token usage for this request: {token_usage}")

    except requests.exceptions.HTTPError as e:
        description = f"HTTP error occurred: {e}"
    except (KeyError, IndexError, TypeError):
        description = "Error: Unable to process image."
    return description


def format_description(description):
    # Replace all periods with commas, convert to lowercase, and remove trailing comma or period
    formatted_description = description.replace(".", ",").lower().rstrip(",.")
    return formatted_description


def image_to_base64(
    image_path="images", image_longest_side_in_px=None, upscale_small_images=True
):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image"):
        raise ValueError("The file type is not recognized as an image")

    with Image.open(image_path) as img:
        if image_longest_side_in_px and (
            max(img.size) > image_longest_side_in_px or upscale_small_images
        ):
            scale = image_longest_side_in_px / max(img.size)
            new_size = tuple([int(x * scale) for x in img.size])
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img_format = "JPEG" if mime_type == "image/jpeg" else "PNG"
        img.save(buffer, format=img_format)
        encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

    image_base64 = f"data:{mime_type};base64,{encoded_string}"
    return image_base64


def process_images(folder, image_longest_side_in_px, upscale_small_images=True):
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            file_path = os.path.join(folder, filename)
            image_base64 = image_to_base64(
                file_path, image_longest_side_in_px, upscale_small_images
            )
            description = send_image_to_openai_api(image_base64)
            description = format_description(description)
            base_filename = os.path.splitext(filename)[0]
            with open(
                os.path.join(folder, f"{base_filename}_description.txt"), "w"
            ) as txt_file:
                txt_file.write(description)


if __name__ == "__main__":
    process_images(image_folder, image_longest_side_in_px, upscale_small_images)
    print(f"Token usage for this session: {token_usage_summary}")
