import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from construct.utils import set_cuda, image_transformation
from construct.architecture import Encoder_Decoder_Model, Vocabulary, Image_encoder, Attention_Based_Decoder, AttentionLayer
import cv2

# Set the device for computation
device = set_cuda()

# Load the pre-trained model and vocabulary
model = torch.load('model/model.pt', map_location=device)
vocab = torch.load("model/vocab.pth", map_location=device)

image = "test.png"

image = Image.open(image)
image = np.array(image)

# Calculate height to maintain aspect ratio with width of 480
aspect_ratio = image.shape[0] / image.shape[1]
new_height = int(480 * aspect_ratio)
image = cv2.resize(image, (480, new_height))

attentions, caption = model.predict(image, vocab)
caption_text = ' '.join(caption[1:-1])

print("Caption:", caption_text)