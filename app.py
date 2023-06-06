import streamlit as st
from roboflow import Roboflow
from PIL import Image
import numpy as np

rf = Roboflow(st.secrets.api_key)
project = rf.workspace().project("arabic-sl")
model = project.version(17).model


#UI
st.title("Image Classification with Streamlit")

option = st.selectbox(
    'Select the approach to test the model',
    ('UploadImage', 'CameraInput'))


if option == 'UploadImage':
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        with open("uploaded_image.png", "wb") as f:
            f.write(uploaded_image.read())
            st.success("Image saved successfully!")
        #Read the saved image
        saved_image = st.image("uploaded_image.png", caption='Uploaded Image.', use_column_width=True)

        predicted = model.predict(
            "uploaded_image.png",
            confidence=20, overlap=30).json()
        st.write("Predicted Class:", predicted)
elif option == 'CameraInput':
    st.write("Camera Input")
    picture = st.camera_input("Take a picture")
    if picture is not None:
        img = Image.open(picture)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        img_array=np.array(img)
        img_array.resize((640, 640))
        #picture = picture.thumbnail((640, 640))
        predicted = model.predict(img_array,
            confidence=20, overlap=30).json()
        st.write("Predicted Class:", predicted)
