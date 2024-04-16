import os

import PIL
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from flask import Flask, request, render_template, send_from_directory
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists('cropped'):
    os.makedirs('cropped')

size = (224, 224)

def crop_and_save_faces(input_dir='faces', output_dir='cropped', output_size=(224, 224)):
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # Detect face
        boxes, _ = mtcnn.detect(img)

        # If at least one face is detected
        if boxes is not None:
            # Assuming the first face is the target
            box = boxes[0]
            face = img.crop(box)

            # Resize the cropped face
            face = face.resize(output_size, PIL.Image.Resampling.LANCZOS)

            # Save the resized cropped face
            face.save(os.path.join(output_dir, img_name))
        else:
            print(f"No face detected in {img_name}.")

if len(os.listdir('faces')) == 0:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(keep_all=True, device=device)

    print("Cropping and saving faces")
    crop_and_save_faces(output_size=size)

def load_and_vectorize_images(directory='cropped'):
    image_vectors = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with Image.open(filepath) as img:
            # Convert to grayscale and resize
            img = img.convert('L')
            # Convert to numpy array and vectorize
            img_vector = np.array(img).flatten()
            image_vectors.append(img_vector)
    return np.array(image_vectors)

def save_centered_vectors(centered_vectors, filename='centered_vectors.npz'):
    np.savez_compressed(filename, centered_vectors=centered_vectors)

def load_centered_vectors(filename='centered_vectors.npz'):
    data = np.load(filename)
    return data

if not os.path.exists('centered_vectors.npz'):
    image_vectors = load_and_vectorize_images()

    mean_face_vector = np.mean(image_vectors, axis=0)
    mean_face_image = mean_face_vector.reshape(size)

    centered_vectors = image_vectors - mean_face_vector
    save_centered_vectors(centered_vectors)
else:
    centered_vectors = load_centered_vectors()

from sklearn.decomposition import PCA

# Fit PCA
n_components = len(centered_vectors)
pca = PCA(n_components=n_components, whiten=True)
pca.fit(centered_vectors)
print('PCA has been fit')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return process_and_show_eigenface(file_path)
    return render_template('upload.html')

def process_and_show_eigenface(file_path):
    img = Image.open(file_path).convert('RGB')
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        box = boxes[0]
        face = img.crop(box)
        face = face.resize(size, PIL.Image.Resampling.LANCZOS)
        img_vector = np.array(face.convert('L')).flatten()
        img_centered = img_vector - mean_face_vector
        eigenface_projection = pca.transform([img_centered])
        eigenface_image = eigenface_projection.dot(pca.components_).reshape(size)
        plt.imsave('static/eigenface.jpg', eigenface_image, cmap='gray')
        return send_from_directory('static', 'eigenface.jpg')
    return "No face detected"

if __name__ == '__main__':
    app.run(debug=True)
