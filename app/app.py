import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import io
from utils import  load_image, extract_face, verify
from model_util import load_model

#Set the layout to the streamlit app as wide
st.set_page_config(
    page_title="FaceID",
    page_icon="🗿",
    layout="wide")

st.title("Facial Verification Application")

#Setup the sidebar
with st.sidebar:
    st.title("FaceID")
    st.image(load_image("app/logo.png"), width=150,)
    st.header("Input:")
    options = ["Webcam", "Local file", "URL", "Testing image"]
    selected_option = st.selectbox("", options)

    st.header("Parameters")
    detection_threshold = st.slider('Detection Threshold', min_value =0.0, max_value = 1.0, value = 0.8)
    verification_threshold = st.slider('Verification Threshold', min_value =0.0, max_value = 1.0, value = 0.8)

    st.info("Detection Threshold: The minimum confidence required to consider two pictures as the same person")
    st.info("Verification Threshold: The minimum percentage of pictures that need to be the same person to consider the face verified")

# Generate two columns
col1, col2 = st.columns(2)
siamese_model = load_model()

if selected_option == "Webcam":
    with col1:
        webcam_img = st.camera_input("Take a picture", label_visibility="hidden")

        if webcam_img is not None:
            bytes_data = webcam_img.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            try:    
                with st.status("Extracting face..."):
                    face = extract_face(cv2_img)
                im = Image.fromarray(face[:,:,::-1])
                im.save("application_data/input_image/input_image.jpg")
                with st.status("Running verification model..."):
                    _, verified = verify(siamese_model, detection_threshold, verification_threshold)
                
                if verified:
                        st.success("Face verified! Access granted.")
                else:
                    st.error("Face not recognized. Access denied.")

                with col2:
                    st.info("This is what the machine learning model takes as input when making a prediction:")
                    st.image(load_image("application_data/input_image/input_image.jpg"))
                    st.info("This picture is then compared to 50 reference images of this kind:")
                    st.image(load_image("app/verification_faces.jpg"))
            except:
                st.error("No face detected in the image. Please try another image.")
                 
if selected_option == "Local file":

    with col1:
        uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "png"], label_visibility="hidden")

        if uploaded_img is not None:
            im = Image.open(uploaded_img)
            image = np.array(im)
            try:
                with st.status("Extracting face..."):
                    face = extract_face(image)
                im = Image.fromarray(face)
                im.save("application_data/input_image/input_image.jpg")
                with st.status("Running verification model..."):
                    _, verified = verify(siamese_model, detection_threshold, verification_threshold)
                if verified:
                        st.success("Face verified! Access granted.")
                else:
                    st.error("Face not recognized. Access denied.")
                with col2:
                    st.info("This is what the machine learning model takes as input when making a prediction:")
                    st.image(load_image("application_data/input_image/input_image.jpg"))
                    st.info("This picture is then compared to 50 reference images of this kind:")
                    st.image(load_image("app/verification_faces.jpg"))
                    
            except:
                st.error("No face detected in the image. Please try another image.")
            
if selected_option == "Testing image":
    with col1:
        im = Image.open("app/issam.jpg")
        image = np.array(im)
        st.image(image, width=400, caption="Testing image")
        
        with col2:
            with st.status("Extracting face..."):
                face = extract_face(image)
            im = Image.fromarray(face)
            im.save("application_data/input_image/input_image.jpg")
            with st.status("Running verification model..."):
                _, verified = verify(siamese_model, detection_threshold, verification_threshold)
            if verified:
                    st.success("Face verified! Access granted.")
            else:
                st.error("Face not recognized. Access denied.")
            st.info("This is what the machine learning model takes as input when making a prediction:")
            st.image(load_image("application_data/input_image/input_image.jpg"))
            st.info("This picture is then compared to 50 reference images of this kind:")
            st.image(load_image("app/verification_faces.jpg"))

if selected_option == "URL":
    with col1:

        image_url = st.text_input("Enter URL")
        if image_url:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            image_array = np.array(image)

            try:
                with st.status("Extracting face..."):
                    face = extract_face(image_array)
                im = Image.fromarray(face)
                im.save("application_data/input_image/input_image.jpg")
                with st.status("Running verification model..."):
                    _, verified = verify(siamese_model, detection_threshold, verification_threshold)
                if verified:
                        st.success("Face verified! Access granted.")
                else:
                    st.error("Face not recognized. Access denied.")
                with col2:
                    st.info("This is what the machine learning model takes as input when making a prediction:")
                    st.image(load_image("application_data/input_image/input_image.jpg"))
                    st.info("This picture is then compared to 50 reference images of this kind:")
                    st.image(load_image("app/verification_faces.jpg"))
                    
            except:
                st.error("No face detected in the image. Please try another image.")
