import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import streamlit as st
import cv2
import re
import string
import random
import os

# Define global constants
image_input_size = (224, 224)
mean_normalization_vec = [0.485, 0.456, 0.406]
std_normalization_vec = [0.229, 0.224, 0.225]

# Define image transformation pipeline
image_transformation = T.Compose([
    T.Resize(image_input_size),
    T.ToTensor(),
    T.Normalize(mean=mean_normalization_vec, std=std_normalization_vec)
])

def set_cuda():
    """
    Set the PyTorch destination device to CUDA if available and print the device in use.

    Returns:
        torch.device: The device in use, either 'cuda' or 'cpu'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use: {}".format(device))
    return device

def detensorize_image(image_tensor, mean_vector, std_vector, denormalize=True):
    """
    Convert a normalized image tensor back to a denormalized format for visualization.

    Args:
        image_tensor (torch.Tensor): The input image tensor with shape (C, H, W).
        mean_vector (list): The mean values used during normalization, usually of length 3 for RGB channels.
        std_vector (list): The standard deviation values used during normalization, usually of length 3 for RGB channels.
        denormalize (bool): If True, denormalizes the tensor using the provided mean and std vectors.

    Returns:
        numpy.ndarray: The detensorized image array with shape (H, W, C).
    """
    # Reshape mean and std vectors to tensors with appropriate shape
    mean_tensor = torch.tensor(mean_vector).view(3, 1, 1)
    std_tensor = torch.tensor(std_vector).view(3, 1, 1)

    if denormalize:
        detensorized_image = (image_tensor * std_tensor) + mean_tensor
    else:
        detensorized_image = image_tensor

    # Convert image dimensions from (C, H, W) to (H, W, C)
    detensorized_image = detensorized_image.permute(1, 2, 0)

    # Convert the image tensor to a numpy array for visualization
    detensorized_image = detensorized_image.numpy()

    return detensorized_image

def standarize_text(text):
    """
    Convert captions into a unified format by removing punctuation, trimming whitespace, and converting to lowercase.

    Args:
        text (str): A single input text string to be standardized.

    Returns:
        str: A standardized text string, stripped of punctuation and converted to lowercase.
    """
    text = text.strip()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = text.lower()
    return text

def plot_attention(img, caption, attention_vectors, is_streamlit=False):
    """
    Plot attention maps on the image in a grid layout with 4 columns.

    Args:
        img (numpy.ndarray): The input image in BGR format.
        caption (list): The caption corresponding to the image.
        attention_vectors (list): A list of attention vectors for each word in the caption.
        is_streamlit (bool): Flag to indicate if the plot is for Streamlit. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The figure containing the attention plots.
    """
    plt.clf()

    temp_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    num_attentions = len(caption) - 1
    num_rows = (num_attentions + 3) // 4  # Calculate number of rows for a 4-column grid

    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows))
    axes = axes.flatten()

    for idx in range(len(axes)):
        if idx < num_attentions:
            temp_att = attention_vectors[idx].reshape(7, 7)
            att_resized = cv2.resize(temp_att, (temp_image.shape[1], temp_image.shape[0]))

            axes[idx].imshow(temp_image)
            axes[idx].imshow(att_resized, cmap="jet", alpha=0.4)
            axes[idx].set_title(f"Word: {caption[idx]}")
            axes[idx].axis("off")
        else:
            axes[idx].axis("off")  # Hide any unused subplots

    plt.tight_layout()
    return fig

def pick_random_image(directory):
    """
    Select a random image file from a specified directory.

    Args:
        directory (str): The directory path containing the images.

    Returns:
        str: The file name of the randomly selected image.

    Raises:
        AssertionError: If no images are found in the directory.
    """
    files = os.listdir(directory)
    images = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
    assert images, "No images found in the directory."
    random_image = random.choice(images)
    return random_image
