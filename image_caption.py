import base64
import io
import mimetypes
import os
import sys
import time

import openai
import requests
from dotenv import load_dotenv
from PIL import Image

# Settings for processing images
image_folder = "images"
image_longest_side_in_px = 768  # Max length for the longer side of the image
save_scaled_image = "none"  # Options: 'overwrite', 'new_file', 'none'
upscale_small_images = True  # Upscale images smaller than the longest side
max_tokens_per_request = 400  # Max tokens per request
request_per_minute = 20  # Requests per minute
max_request_retries = 5  # Maximum number of retries for failed requests
overwrite_existing_text = False  # Whether to overwrite existing text files

# Set OpenAI API key in .env file
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("No OpenAI API key provided. Please set the OPENAI_API_KEY in the .env file.")
    sys.exit(1)

openai.api_key = openai_api_key

# Global token usage summary
token_usage_summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def send_image_to_openai_api(image_base64, rpm=30, max_retries=5):
    delay = 60 / rpm
    retry_count = 0

    while retry_count < max_retries:
        print(
            f"Attempt {retry_count + 1}: Delay between requests: {delay} seconds."
        )
        time.sleep(delay)

        instructions = (
            "Generate a concise, descriptive caption for the image, focusing on key elements and features. "
            "Analyze all elements in the image, understanding both the overall composition and individual components. "
            "Describe elements using single words or brief phrases, avoiding long sentences, and separate them with commas. "
            "Include tags for poses, orientations, and background styles where relevant. "
            "Identify the most important elements in images, such as the style (e.g., modern, futuristic, elegant), color scheme, and other significant aspects of the image. "
            "For user interfaces, describe UI elements in more generic terms (e.g., 'image gallery', 'product image') rather than detailing their specific content. "
            "Refrain from using specific names of locations, brands, or individuals in your descriptions. Replace these with generic terms that broadly categorize the subject. "
            "Example: For an image with an in-vehicle infotainment system, use 'user interface, clean design, navigation, music player, in-vehicle infotainment, white style, button bar on bottom.' "
        )
        headers = {"Authorization": f"Bearer {openai_api_key}"}
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": image_base64,
                }
            ],
            "max_tokens": max_tokens_per_request,
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            response_data = response.json()
            description = response_data["choices"][0]["message"]["content"]

            token_usage = response_data.get("usage", {})
            global token_usage_summary
            token_usage_summary["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
            token_usage_summary["completion_tokens"] += token_usage.get(
                "completion_tokens", 0
            )
            token_usage_summary["total_tokens"] += token_usage.get("total_tokens", 0)
            print(f"Token usage for this request: {token_usage}")

            return description

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            if e.response.status_code == 429:
                retry_count += 1
                delay *= 2
                continue
            else:
                description = "Error: Unable to process image."
                return description
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            retry_count += 1
            delay *= 2
            continue

    print("Maximum retries reached. Exiting.")
    return "Error: Maximum retries reached."


def format_description(description):
    formatted_description = description.replace(".", ",").lower().rstrip(",.")
    return formatted_description


def image_to_base64(
    image_path,
    image_longest_side_in_px=None,
    upscale_small_images=True,
    save_scaled_image="none",
):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image"):
        raise ValueError("The file type is not recognized as an image")

    with Image.open(image_path) as img:
        original_size = img.size
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

        if save_scaled_image != "none" and new_size != original_size:
            if save_scaled_image == "overwrite":
                img.save(image_path, format=img_format)
            elif save_scaled_image == "new_file":
                new_image_path = (
                    os.path.splitext(image_path)[0]
                    + "_scaled"
                    + os.path.splitext(image_path)[1]
                )
                img.save(new_image_path, format=img_format)

    image_base64 = f"data:{mime_type};base64,{encoded_string}"
    return image_base64


def process_images(
    folder,
    image_longest_side_in_px,
    upscale_small_images=True,
    rpm=30,
    max_retries=5,
    save_scaled_image="none",
    overwrite_existing_text=False,
):
    image_files = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]
    total_images = len(image_files)
    images_processed = 0

    for filename in image_files:
        file_path = os.path.join(folder, filename)
        base_filename = os.path.splitext(filename)[0]
        text_file_path = os.path.join(folder, f"{base_filename}.txt")

        if not overwrite_existing_text and os.path.exists(text_file_path):
            print(f"Skipping {filename} as text file already exists.")
            images_processed += 1
            continue

        try:
            image_base64 = image_to_base64(
                file_path, image_longest_side_in_px, upscale_small_images, save_scaled_image
            )
            description = send_image_to_openai_api(image_base64, rpm)
            description = format_description(description)
            
            with open(text_file_path, "w") as txt_file:
                txt_file.write(description)

            images_processed += 1
            remaining_images = total_images - images_processed
            estimated_remaining_time = (remaining_images * 60) / rpm
            print(
                f"Processed {images_processed}/{total_images}. Estimated remaining time: {estimated_remaining_time:.2f} seconds."
            )
        except Exception as e:
            print(f"Failed to process image {filename}: {e}")
            continue


if __name__ == "__main__":
    process_images(
        folder=image_folder,
        image_longest_side_in_px=image_longest_side_in_px,
        upscale_small_images=upscale_small_images,
        rpm=request_per_minute,
        max_retries=max_request_retries,
        save_scaled_image=save_scaled_image,
        overwrite_existing_text=overwrite_existing_text,
    )
    print(f"Session ended. Token usage summary: {token_usage_summary}")
