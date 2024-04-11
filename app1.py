from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define the upload image route
@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']

        # Save the image
        image.save('uploaded_image.jpg')

        # Preprocess the image
        image = Image.open('uploaded_image.jpg').resize((224, 224))
        image = np.array(image) / 255.0

        # Make a prediction
        prediction = model.predict(np.expand_dims(image, axis=0))[0]

        # Get the predicted class
        predicted_class = np.argmax(prediction)

        # Render the template with the prediction
        return render_template('index.html', prediction=predicted_class)

    return render_template('index.html')

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)