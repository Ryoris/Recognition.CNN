from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create the folder if it doesn't exist

# Initialize the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

def predict_image(image_path):
    """
    Process the image and make a prediction using the InceptionV3 model.
    :param image_path: Path to the image file
    :return: List of top 5 predictions
    """
    img = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
    img = img.resize((299, 299))  # Resize image to the size expected by InceptionV3
    x = np.expand_dims(np.array(img), axis=0)  # Convert image to numpy array and add batch dimension
    x = preprocess_input(x)  # Preprocess image
    predictions = model.predict(x)  # Make prediction
    decoded_predictions = decode_predictions(predictions, top=5)[0]  # Decode predictions
    return decoded_predictions

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve the uploaded file from the uploads folder.
    :param filename: Name of the file to be served
    :return: File from the uploads folder
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload():
    """
    Handle the upload of an image and return the filename as JSON.
    :return: Render index.html on GET, return filename JSON on POST
    """
    if request.method == 'POST':
        file = request.files['file']  # Get the uploaded file
        if file:
            filename = secure_filename(file.filename)  # Secure the filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Create the full filepath
            file.save(filepath)  # Save the file
            return jsonify({'filename': filename})  # Return the filename as JSON
    return render_template('index.html')  # Render the upload form on GET

@app.route('/result/<filename>', methods=['GET'])
def result(filename):
    """
    Display the prediction results for the uploaded image.
    :param filename: Name of the uploaded file
    :return: Render result.html with predictions and image path
    """
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Get the full path of the file
    predictions = predict_image(filepath)  # Make predictions
    return render_template('result.html', predictions=predictions, filename=filename)  # Render result page

if __name__ == '__main__':
    app.run(debug=True)
