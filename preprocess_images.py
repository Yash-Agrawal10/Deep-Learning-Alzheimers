import os
import numpy as np
import pickle
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_and_preprocess_images(categories):
    # Prepare data and labels
    data = []
    labels = []

    # Loop through each category
    for category_id, category in enumerate(categories):
        for image_name in os.listdir(category):
            # Read the image
            image = load_img(os.path.join(category, image_name))
            # Convert the image to numpy array
            image = img_to_array(image)
            # Normalize the image
            image = image.astype('float32') / 255.0
            # Append the image and its corresponding label to the data and labels list
            data.append(image)
            labels.append(category_id)

    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def split_data(data, labels):
    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    return train_data, test_data, train_labels, test_labels

def save_data(data, filename):
    # Save the preprocessed data
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    # Load the preprocessed data
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Define categories and data size
categories = ['Data/Mild Dementia', 'Data/Moderate Dementia', 'Data/Non Demented', 'Data/Very mild Dementia']

# Load and preprocess images
data, labels = load_and_preprocess_images(categories)

# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = split_data(data, labels)

# Save data
save_data(train_data, 'Data/train_data.pkl')
save_data(test_data, 'Data/test_data.pkl')
save_data(train_labels, 'Data/train_labels.pkl')
save_data(test_labels, 'Data/test_labels.pkl')