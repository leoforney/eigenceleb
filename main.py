import os

import PIL
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
import torch

# Ensure the 'cropped' directory exists
if not os.path.exists('cropped'):
    os.makedirs('cropped')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

mtcnn = MTCNN(keep_all=True, device=device)

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

#crop_and_save_faces(output_size = size)

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

image_vectors = load_and_vectorize_images()

mean_face_vector = np.mean(image_vectors, axis=0)
mean_face_image = mean_face_vector.reshape(size)

from sklearn.decomposition import PCA

# Center the data by subtracting the mean face
centered_vectors = image_vectors - mean_face_vector

# Fit PCA
n_components = len(image_vectors)
pca = PCA(n_components=n_components, whiten=True)
pca.fit(centered_vectors)

# The principal components are the eigenfaces
eigenfaces = pca.components_.reshape((n_components, *size))

from matplotlib import pyplot as plt

plt.imshow(mean_face_image, cmap='gray')
plt.title("Average Face")
plt.show()

plt.imsave('eigenface.jpg', eigenfaces[0])