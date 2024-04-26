import os
import numpy as np
import random
from keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_images(categories, num_samples):
    # Prepare data and labels
    data = []
    labels = []

    # Loop through each category
    for category_id, category in enumerate(categories):
        # Get a list of all images in the category
        all_images = [img for img in os.listdir(category) if img.endswith('.jpg')]
        # Randomly select num_samples images
        selected_images = random.sample(all_images, num_samples)

        for image_name in selected_images:
            # Read the image
            image = load_img(os.path.join(category, image_name))
            # Convert the image to numpy array
            image = img_to_array(image)
            # Normalize the image
            image = image.astype('float32') / 255.0
            # Append the image and its corresponding label to the data and labels list
            data.append(image)
            labels.append(category_id)

    return np.array(data), np.array(labels)

def save_data(data, filename):
    # Save the preprocessed data
    np.save(filename, data)

def load_data(filename):
    # Load the preprocessed data
    return np.load(filename)

# Define categories and number of samples
categories = ['Data/Mild Dementia', 'Data/Moderate Dementia', 'Data/Non Demented', 'Data/Very mild Dementia']
num_samples = 488

# Load and preprocess images
data, labels = load_and_preprocess_images(categories, num_samples)

# Save data
save_data(data, 'Data/data.npy')
save_data(labels, 'Data/labels.npy')