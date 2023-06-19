import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.model import build_model
from src import config

dataset_classes = config.CLASS_NAMES
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'webapp/static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(pretrained=False, fine_tune=False, num_classes=3)
checkpoint = torch.load('outputs/model.pth', map_location=config.DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

# Transformations to apply to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        return redirect(url_for('predict', filename=filename))

    return redirect(url_for('index'))

@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Open and preprocess the image
    image = Image.open(file_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make the prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Get the class name
    class_name = dataset_classes[predicted.item()]
    return render_template('result.html', filename=filename, class_name=class_name)

@app.route('/performance')
def performance():
    # Create the confusion matrix figure path
    confusion_matrix_path = 'outputs/confusion_matrix.png'
    # Create the model loss and accuracy plot paths
    model_loss_path = 'outputs/model_loss.png'
    model_accuracy_path = 'outputs/model_accuracy.png'

    return render_template('performance.html', confusion_matrix_path=confusion_matrix_path,
                           model_loss_path=model_loss_path, model_accuracy_path=model_accuracy_path)


if __name__ == '__main__':
    app.run(debug=True)
