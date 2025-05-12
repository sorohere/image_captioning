import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import streamlit as st
import cv2
import re
import string
import random
import os

# Image processing constants
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
    Set PyTorch to use CUDA if available.

    Returns:
        torch.device: The device in use, either 'cuda' or 'cpu'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")
    return device

def detensorize_image(image_tensor, mean_vector, std_vector, denormalize=True):
    """
    Converts a normalized image tensor back to a denormalized format for visualization.

    Args:
        image_tensor (torch.Tensor): The input image tensor with shape (C, H, W).
        mean_vector (list): The mean values used during normalization, usually of length 3 for RGB channels.
        std_vector (list): The standard deviation values used during normalization, usually of length 3 for RGB channels.
        denormalize (bool): If True, denormalizes the tensor using the provided mean and std vectors.

    Returns:
        numpy.ndarray: The detensorized image array with shape (H, W, C).
    """
    mean_tensor = torch.tensor(mean_vector).view(3, 1, 1)
    std_tensor = torch.tensor(std_vector).view(3, 1, 1)
    if denormalize:
        image_tensor = (image_tensor * std_tensor) + mean_tensor
    image_array = image_tensor.permute(1, 2, 0).numpy()
    return image_array

def standarize_text(text):
    """
    Standardize text by removing punctuation, trimming whitespace, and converting to lowercase.

    Args:
        text (str): The input text string to be standardized.

    Returns:
        str: A standardized text string.
    """
    text = text.strip()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    return text.lower()

def plot_attention(img, caption, attention_vectors, is_streamlit=False):
    """
    Display attention heatmaps for each word in the caption.

    Args:
        img (numpy.ndarray): The input image with shape (H, W, C).
        caption (list[str]): List of words in the caption corresponding to the attention vectors.
        attention_vectors (list[numpy.ndarray]): Attention scores for each word, each with a flattened shape of (49,).
        is_streamlit (bool): If True, displays the plot using Streamlit's st.pyplot.

    Returns:
        None
    """
    assert len(caption) > 1, f"Caption has only {len(caption)} words."

    fig, axes = plt.subplots(len(caption) - 1, 1, figsize=(8, 8 * len(caption)))

    for l, ax in enumerate(axes):
        temp_att = attention_vectors[l].reshape(7, 7)  # Reshape attention vector to 7x7 map
        att_resized = cv2.resize(temp_att, (img.shape[1], img.shape[0]))  # Resize attention map to match image size
        ax.imshow(img)
        ax.imshow(att_resized, cmap='jet', alpha=0.4)  # Overlay heatmap on the image
        ax.set_title(caption[l])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    if is_streamlit:
        st.pyplot(fig)

def pick_random_image(directory):
    """
    Select a random image file from a specified directory.

    Args:
        directory (str): Path to the directory containing image files.

    Returns:
        str: The file name of the randomly selected image.
    """
    files = os.listdir(directory)
    images = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
    assert images, "No images found in the directory."
    return random.choice(images)