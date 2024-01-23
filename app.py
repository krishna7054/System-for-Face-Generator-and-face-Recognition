from flask import Flask, render_template, request, redirect, url_for
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

app = Flask(__name__ , static_url_path='/static')

# Load the generator model
generator = load_model('./out/krish_face_generator')

# Create an MTCNN face detector
detector = MTCNN()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_and_detect', methods=['POST'])
def generate_and_detect():
    # Generate images
    num_generated_images = 1
    latent_vectors = np.random.normal(size=(num_generated_images, 32))
    generated_images = generator.predict(latent_vectors)
    generated_images = (generated_images * 0.5 + 0.5) * 255
    generated_image_path = "static/generated_image.png"
    Image.fromarray(generated_images[0].astype(np.uint8)).save(generated_image_path)

    # Detect faces in the generated image
    image = cv2.imread(generated_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = detector.detect_faces(image_rgb)

    # Draw rectangles around the faces
    for face in detected_faces:
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)

    # Save the image with detected faces
    detected_image_path = "static/detected_image.png"
    Image.fromarray(image).save(detected_image_path)

    # Display the result
    return render_template('result.html', generated_image=generated_image_path, detected_image=detected_image_path)

if __name__ == '__main__':
    app.run(debug=True)
