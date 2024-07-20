import streamlit as st
import subprocess
from random import randint
import numpy as np
import pickle
import streamlit as st
import base64
import streamlit as st

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:background/background.avif;base64,%s");
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
st.title("VOICE ENABLED GESTURE INTERPRETATON")

# Create a dropdown menu for options
options = ['None','VideoGenerated', 'Frameconversion', 'prediction']
selected_option = st.selectbox('Select an option', options)

if selected_option == 'None':
    # Code for the Translate feature
    st.title('Welcome to Gesture Page ')

elif selected_option == 'VideoGenerated':
    # Code for the Legal Document feature
    st.title('VideoGenerated')
    #subprocess.run(["python", "Video.py"])
    subprocess.run(["streamlit", "run", "app4.py"])


elif selected_option == 'Frameconversion':
    # Code for the Doubt feature
    st.title('Frameconversion')
    #subprocess.run(["python", "app5.py"])
    subprocess.run(["streamlit", "run", "app5.py"])


elif selected_option == 'prediction':
    # Code for the Translate feature
    st.title('prediction')
    subprocess.run(["streamlit", "run", "app2.py"]) 


