#read api
import dotenv
import os
import requests
from dotenv import load_dotenv
from PIL import Image
import io

# load_dotenv()
# hf_token=os.getenv("hf_token")

# API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
# headers = {"Authorization": f"Bearer {hf_token}"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.content

# def saveimage(prompt):
#     image_bytes = query({
#         "inputs": prompt,
#     })

#     image = Image.open(io.BytesIO(image_bytes))
#     image.save("genfromapi.jpg")


def generate_and_save_image(prompt, filename="genfromapi.jpg"):
    load_dotenv()
    hf_token = os.getenv("hf_token")
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code == 200:
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        image.save(filename)
        print(f"Image saved as {filename}")
    else:
        print(f"Error: {response.status_code} - {response.text}")