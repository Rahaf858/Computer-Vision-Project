import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
import os

#load the model 
model_path = r"C:\Users\Rahsh\OneDrive\Desktop\Rahaf_Fatma_Lama_Project\Rahaf_Fatma_Lama_Project\Project_Files\Handwritten_Digit\model.h5"
model = tf.keras.models.load_model(model_path)

canvas_width = 300
canvas_height = 300
stroke_width = 10
drawing_mode = st.sidebar.selectbox("Drawing mode", ["freedraw"])

def preprocess_image(img):
    img = cv2.resize(255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255
    return img

def predict_digit(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    digit_index = np.argmax(prediction)
    digit_str = str(digit_index)  # Convert digit index to string
    return digit_str

def main():
    st.title("Handwritten Digit Recognition")
    canvas = st_canvas(
        fill_color="white",
        stroke_width=stroke_width,
        stroke_color="black",
        background_color="white",
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    if st.button("Predict"):
        drawn_image = canvas.image_data.astype(np.uint8)[..., :3]
        preprocessed_image = preprocess_image(cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR))
        prediction = model.predict(preprocessed_image)
        digit_index = np.argmax(prediction)
        digit_str = str(digit_index)
        st.markdown(f"*Predicted Digit: {digit_str}*")
        st.image(drawn_image, caption="Drawn Image")


if __name__ == "__main__":
    main()
