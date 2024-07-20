import cv2
import os
import streamlit as st
import tempfile
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import base64
from gtts import gTTS
import os
# Function to get base64 encoding of a file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown('<style>h1 { color: White ; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background/3.jpg')

# Streamlit app title
st.title("VOICE ENABLED GESTURE INTERPRETATION")
import streamlit as st
import cv2
import numpy as np
import tempfile

st.title("Four Images to Video Converter")

uploaded_files = st.file_uploader("Choose exactly four images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != 4:
        st.error("Please upload exactly four images.")
    else:
        images = []
        for uploaded_file in uploaded_files:
            # Load each image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            images.append(image)
        
        # Display the images
        st.image([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images], width=150)
        
        # Get dimensions of the first image
        height, width, layers = images[0].shape
        
        # Create a temporary file to save the video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_name = temp_file.name

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

        # Write each image as a frame in the video
        for image in images:
            video.write(image)
        video.release()

        # Provide download link for the video
        st.success("Video created successfully!")
        with open(video_name, "rb") as video_file:
            st.download_button(label="Download Video", data=video_file, file_name="output.mp4", mime="video/mp4")
