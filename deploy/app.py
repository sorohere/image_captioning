import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import streamlit as st
from construct.utils import set_cuda, image_transformation, plot_attention
from construct.architecture import Encoder_Decoder_Model, Vocabulary, Image_encoder, Attention_Based_Decoder, AttentionLayer
import numpy as np
from PIL import Image

# Set the device for computation
device = set_cuda()

# Load the pre-trained model and vocabulary
model = torch.load('deploy/model/model.pt', map_location=device)
vocab = torch.load("deploy/model/vocab.pth", map_location=device)

# Configure Streamlit app with a dark theme and page layout
st.set_page_config(
    page_title="capage",
    layout="centered",  # Center content for better aesthetics
    initial_sidebar_state="expanded"
)

# CSS styles for dark theme and improved visuals
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #E0E0E0;
        }
        .title {
            text-align: center;
            font-size: 36px;
            color: #BB86FC;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .header {
            text-align: center;
            font-size: 18px;
            color: #CF6679;
            margin-bottom: 20px;
        }
        .stButton button {
            color: #fff; /* Change font color to white */
            font-size: 16px;
            font-weight: bold; /* Make font bold */
            padding: 0px 10px;
            border-radius: 8px;
            display: block;
            margin: 20px auto;
        }

        .stButton button:hover {
            color: #FF0000; /* Change text color to red */
            border: 2px solid #FF0000; /* Add red border */
        }

        .matrix-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
            max-width: 480px;
            margin: 0 auto;
        }
        .image-container {
            width: 100%;
            height: auto;
            border-radius: 8px;
            overflow: hidden;
            margin: 0 auto;
        }
        .image-container img {
            width: 100%;
            height: auto;
            object-fit: contain;
        }
        .textbox-container {
            width: 100%;
            display: flex;
            justify-content: center;
            margin-bottom: 30px; /* Add space below the text box */
        }
        .textbox {
            width: 100%;
            height: 100px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #BB86FC;
            font-size: 16px;
            background-color: #1F1B24;
            color: #E0E0E0;
        }
        /* Center the file uploader */
        .st-emotion-cache-1v0mbdj {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Display title and instructions
st.markdown('<div class="title">capage: imagecaption</div>', unsafe_allow_html=True)
st.markdown('<div class="header">upload an image and watch the captioning model work its brain!</div>', unsafe_allow_html=True)

# Upload image section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Load and preprocess the uploaded image
    image = Image.open(uploaded_image)
    image = np.array(image)
    
    # Calculate height to maintain aspect ratio with width of 480
    aspect_ratio = image.shape[0] / image.shape[1]
    new_height = int(480 * aspect_ratio)
    image = cv2.resize(image, (480, new_height))

    # Wrap image in container div
    st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Button to generate captions
    if st.button("generate caption"):
        with st.spinner("Generating caption..."):
            try:
                # Generate caption and attention maps
                attentions, caption = model.predict(image, vocab)
                caption_text = ' '.join(caption[1:-1])

                # Store caption and attentions in session state
                st.session_state.caption = caption_text
                st.session_state.attentions = attentions
                st.session_state.full_caption = caption

            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")

    # Show caption text box if a caption is generated
    if "caption" in st.session_state:
        # Display the generated caption in a text box
        st.markdown(
            f'<div class="textbox-container"><textarea class="textbox" readonly>{st.session_state.caption}</textarea></div>',
            unsafe_allow_html=True
        )

        # Show checkbox for attention maps
        show_attention = st.checkbox("show attentions", value=False)

        # Display attention maps based on checkbox state
        if show_attention:
            try:
                # Create a placeholder for attention maps at the very end
                attention_placeholder = st.empty()
                fig = plot_attention(image, st.session_state.full_caption, st.session_state.attentions, is_streamlit=True)
                attention_placeholder.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error displaying attention maps: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Upload an image to get started!")


