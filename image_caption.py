import base64
import os
import subprocess
import sys
import mimetypes
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now you can use the environment variable as before
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    print("No OpenAI API key provided. Please set the OPENAI_API_KEY in the .env file.")
    sys.exit(1)

openai.api_key = openai_api_key


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


def send_image_to_openai_api(image_base64):
    # Send the image to OpenAI API with detailed instructions for generating captions
    instructions = (
        "Generate a concise, descriptive caption for the image, focusing on key elements and features. "
        "Analyze all elements in the image, understanding both the overall composition and individual components. "
        "Describe elements using single words or brief phrases, avoiding long sentences, and separate them with commas. "
        "Include tags for poses, orientations, and background styles where relevant. "
        "Example: For an image with an in-vehicle infotainment system, use 'user interface, clean design, navigation, music player, in-vehicle infotainment, white style, button bar on bottom.'"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
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
        max_tokens=2000,
    )
    return response.choices[0].message.content


def process_images(folder):
    # Process each image in the folder
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            file_path = os.path.join(folder, filename)
            image_base64 = image_to_base64(file_path)
            description = send_image_to_openai_api(image_base64)

            # Save the description to a .txt file
            base_filename = os.path.splitext(filename)[0]
            with open(
                os.path.join(folder, f"{base_filename}_description.txt"), "w"
            ) as txt_file:
                txt_file.write(description)


if __name__ == "__main__":
    # Use the script's location as the folder path if not provided by the user
    script_directory = os.path.dirname(os.path.realpath(__file__))
    image_folder = script_directory
    process_images(image_folder)
