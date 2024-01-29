import base64
import mimetypes
import os
import subprocess
import sys

import requests
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
image_folder = "images"

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
        description = response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        description = f"HTTP error occurred: {e}"
    except (KeyError, IndexError, TypeError):
        description = "Error: Unable to process image."
    return description


def format_description(description):
    # Replace all periods with commas and convert to lowercase
    return description.replace(".", ",").lower()


def scale_image(image_path, max_length, upscale_smaller_images=True):
    with Image.open(image_path) as img:
        # Only scale if the image is larger than max_length or if upscaling is allowed
        if max(img.size) > max_length or (
            upscale_smaller_images and max(img.size) < max_length
        ):
            scale = max_length / max(img.size)
            new_size = tuple([int(x * scale) for x in img.size])
            img = img.resize(new_size, Image.ANTIALIAS)
            img.save(image_path)


def image_to_base64(image_path):
    # Guess the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image"):
        raise ValueError("The file type is not recognized as an image")
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    # Format the result with the appropriate prefix
    image_base64 = f"data:{mime_type};base64,{encoded_string}"
    return image_base64


def process_images(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            file_path = os.path.join(folder, filename)
            image_base64 = image_to_base64(file_path)
            description = send_image_to_openai_api(image_base64)
            description = format_description(description)
            base_filename = os.path.splitext(filename)[0]
            with open(
                os.path.join(folder, f"{base_filename}_description.txt"), "w"
            ) as txt_file:
                txt_file.write(description)


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.realpath(__file__))
    process_images(image_folder)
