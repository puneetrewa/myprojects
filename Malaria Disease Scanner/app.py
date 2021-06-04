import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model('malaria_cnn.h5')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.write("""
         # MALARIA DISEASE DETECTION
         """
         )
st.write("This is a simple image classification web app to predict the malaria disease (By - SPARL Project Group)")
file = st.file_uploader("Upload an image file", type=["jpg", "png"])



def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(64, 64),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

if file is None:
    st.text("Please upload an image file")

else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) >= 0.5:
        st.write("Malaria Detected!")
    else:
        st.write("Malaria not detected ")
    
    st.text("Chance of Malaria (1:Parasitized, 0:Uninfected)")
    st.write(prediction)