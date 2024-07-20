# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:10:52 2024

@author: sribr
"""

import cv2
import streamlit as st
import tempfile
import os
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
# Set up the Streamlit interface
st.title("Video to Frame Extractor")

# Upload video file
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Open the video file
    cap = cv2.VideoCapture(temp_file_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video.")
    else:
        st.write("Video frame Converted complete.")
        # Initialize frame count
        frame_count = 0

        stframe = st.empty()

        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                break

            # Display the frame in the Streamlit app
            stframe.image(frame, channels="BGR")
           
            

            # Save the frame as an image (optional)
            cv2.imwrite(f'frame_{frame_count}.jpg', frame)
           # Save frame as an image file

            # Increment frame count
            frame_count += 1

            # Limit the frame rate to make sure Streamlit can render it
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Release the video capture object
        cap.release()

    # Clean up the temporary file
    os.remove(temp_file_path)


