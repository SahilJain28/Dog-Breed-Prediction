import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

#Load the dog identification model
identification_model = load_model('dog_identification_model.h5')

#Load the dog breed classifier model
breed_model = load_model('dog_breed_class.h5')

CLASS_NAMES =['Scottish Deerhound', 'Maltese Dog','Bernese Mountain Dog']

#Function to predict if the image contains a dog
def predict_dog(image_path):
    img = image.load_img(image_path, target_size = (150,150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = identification_model.predict(img_array)
    return prediction[0][0] > 0.5

st.title("Dog Breed Prediction")
st.markdown("Upload an image of the Dog")


dog_image = st.file_uploader("Choose an image...",type = "jpeg")
submit = st.button('Predict')

if submit:

    if dog_image is None:
        st.title(str("No image uploaded"))    

    if dog_image is not None:
        image_path = "temp_image.jpg"
        
        with open(image_path, "wb") as f:
            f.write(dog_image.getbuffer())

        if predict_dog(image_path):
            file_bytes = np.asarray(bytearray(dog_image.read()),dtype=np.uint8)
            opencv_image =cv2.imdecode(file_bytes,1)
            
            st.image(opencv_image,channels="BGR")
            opencv_image = cv2.resize(opencv_image,(224,224))
            opencv_image.shape =(1,224,224,3)
            Y_pred = breed_model.predict(opencv_image)
            
            st.title(str("The Dog Breed is "+CLASS_NAMES[np.argmax(Y_pred)]))

        else:
            file_bytes = np.asarray(bytearray(dog_image.read()),dtype=np.uint8)
            opencv_image =cv2.imdecode(file_bytes,1)

            st.image(opencv_image,channels="BGR")

            st.title(str("This image is irrelevant.Please upload a dog image!"))