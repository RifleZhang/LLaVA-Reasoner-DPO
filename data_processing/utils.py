import os, os.path as osp
import json
import pickle
import json
import random
import torch
import copy
import glob
import numpy as np
import base64
import pandas as pd
from logzero import logger
from PIL import Image
import io
from io import BytesIO
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Lambda, ToTensor
import re
import boto3

ENDPOINT_URL=os.environ.get("ENDPOINT_URL", "")
BUCKET_NAME=os.environ.get("BUCKET_NAME", "")
DEFAULT_IMAGE_FOLDER=os.environ.get("DEFAULT_IMAGE_FOLDER", "")
DEFAULT_BUCKET_FOLDER=os.environ.get("DEFAULT_BUCKET_FOLDER", "")

def remove_dup(data):
    filtered_data = []
    dic = set()
    for x in data:
        if x['id'] in dic:
            continue
        filtered_data.append(x)
        dic.add(x['id'])
    return filtered_data
    
def count(data):
    return len(set([_['id'] for _ in data]))

def sample(x, num):
    return list(np.random.choice(x, num, replace=False))

def get_id_from_frame_path(path):
    return path.split('/')[-1].split('.')[0]

def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# ---------------------------- <image loading and processing> ----------------------------
S3 = boto3.client('s3', endpoint_url=ENDPOINT_URL)

def load_and_cache_image(image_file, bucket_folder=DEFAULT_BUCKET_FOLDER, local_folder=DEFAULT_IMAGE_FOLDER):
    if './' in image_file:
        image_file = image_file.replace('./', '')
    local_image_path = os.path.join(local_folder, image_file)
    try:
        if not os.path.exists(local_image_path):  # Check if image is cached locally
            bucket_image_path = os.path.join(bucket_folder, image_file)
            # Fetch image from S3 bucket
            obj = S3.get_object(Bucket=BUCKET_NAME, Key=bucket_image_path)
            image_data = obj['Body'].read()

            # Create directory if it does not exist
            local_image_dir = os.path.dirname(local_image_path)
            os.makedirs(local_image_dir, exist_ok=True)

            # Write image data to local path
            with open(local_image_path, 'wb') as f:
                f.write(image_data)
            
            # Load image from bytes, convert to RGB to ensure consistency
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            # Load image from local cache, convert to RGB
            img = Image.open(local_image_path).convert('RGB')
    except Exception as e:
        # Handle exceptions that could occur, like S3 access errors or corrupt image files
        print(f"An error occurred:\nlocal image {local_image_path}\nbucket image {bucket_image_path}\n{e}")
        return None
    return img

def maybe_download_image(image_file, bucket_folder=DEFAULT_BUCKET_FOLDER, local_folder=DEFAULT_IMAGE_FOLDER):
    if './' in image_file:
        image_file = image_file.replace('./', '')
    local_image_path = os.path.join(local_folder, image_file)
    bucket_image_path = os.path.join(bucket_folder, image_file)
 
    if not os.path.exists(local_image_path):  # Check if image is cached locally
        # Fetch image from S3 bucket
        obj = S3.get_object(Bucket=BUCKET_NAME, Key=bucket_image_path)
        image_data = obj['Body'].read()

        # Create directory if it does not exist
        local_image_dir = os.path.dirname(local_image_path)
        os.makedirs(local_image_dir, exist_ok=True)

        # Write image data to local path
        with open(local_image_path, 'wb') as f:
            f.write(image_data)

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def image_to_base64(image_path):
    '''
    Converts an image from a specified file path to a base64-encoded string.
    
    Parameters:
    image_path (str): A string representing the file path of the image to be converted.

    Returns:
    str: A base64-encoded string representing the image.
    '''
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_str):
    '''
    Converts a base64-encoded string back to an image object.
    
    Parameters:
    base64_str (str): A base64-encoded string representing an image.

    Returns:
    Image: An image object reconstructed from the base64 string.
    '''
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))

def display_image(image):
    plt.figure(dpi=200)
    if isinstance(image, str):
        image = load_image(image)
    plt.imshow(image, interpolation='none')
    plt.axis('off')  # Turn off axis numbers and labels
    plt.show()
# ---------------------------- </image loading and processing> ----------------------------

# ---------------------------- <text processing> ----------------------------
def load_text(path):
    with open(path, "r") as f:
        text = f.readlines()[0]
    return text

def load_text(path):
    with open(path, "r") as f:
        text = f.readlines()
    return text

def save_text(path, texts, mode='w'):
    if isinstance(texts, list):
        text = '\n'.join(texts)
    else:
        text = texts
    with open(path, mode) as f:
        f.write(text)

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_jsonl(save_path, data, append=False):
    if append:
        mode = "a"
    else:
        mode = "w"
    if type(data) == list:
        with open(save_path, mode) as f:
            for line in data:
                json.dump(line, f)
                f.write("\n")
    else:
        with open(save_path, mode) as f:
            json.dump(data, f)
            f.write("\n")

def load_json_data(path):
    if "jsonl" in path:
        data = load_jsonl(path)
    else:
        data = load_json(path)
    return data

def load_jsonl(save_path):
    with open(save_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def format_docstring(docstring: str) -> str:
    """Format a docstring for use in a prompt template."""
    return re.sub("\n +", "\n", docstring).strip()
# ---------------------------- </text processing> ----------------------------

# ---------------------------- <data loading and processing> ----------------------------
def load_parquet_from_dir(data_dir):
    data_files = glob.glob(osp.join(data_dir, "*.parquet"))
    data_files.sort()
    all_data_chunks = []
    for data_file in data_files:
        data = pd.read_parquet(data_file)
        for idx, row in data.iterrows():
            all_data_chunks.append(row)
    return all_data_chunks
# ---------------------------- </data loading and processing> ----------------------------