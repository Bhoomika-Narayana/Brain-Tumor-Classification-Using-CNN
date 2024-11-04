import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from keras.preprocessing.image import img_to_array

model_path = r"C:\Users\navee\OneDrive\Documents\newproject\Brain_Tumor_Classification.keras"
# Load model without compiling
model = tf.keras.models.load_model(model_path, compile=False)

# Recompile the model with default settings
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class_mappings = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}

def load_and_preprocess_image(image, target_size=(168, 168)):
    img = image.convert('L')  # Convert image to grayscale
    img = img.resize(target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add the batch dimension
    return img_array

def predict_image_class(model, image, class_mappings):
    try:
        preprocessed_img = load_and_preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = 'The image is classified as ' + class_mappings[predicted_class_index] 
        return predicted_class_name
    except Exception as e:
        return f"Error in prediction: {str(e)}"

img1 = Image.open('brain.jpg')
st.image(img1, use_column_width=False)

st.title('Cerebral Neoplasm Classification using Neural Networks')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)  

        with col1:
            resized_img = image.resize((300, 300))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                prediction = predict_image_class(model, image, class_mappings)
                st.success(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"Error in processing the uploaded image: {str(e)}")
