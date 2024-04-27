# import os
# import numpy as np
# import random
# from keras.preprocessing.image import load_img, img_to_array
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical

# def load_and_preprocess_images(categories, num_samples):
#     # Prepare data and labels
#     data = []
#     labels = []

#     # Loop through each category
#     for category_id, category in enumerate(categories):
#         # Get a list of all images in the category
#         all_images = [img for img in os.listdir(category) if img.endswith('.jpg')]
#         # Randomly select num_samples images
#         selected_images = random.sample(all_images, num_samples)

#         for image_name in selected_images:
#             # Read the image
#             image = load_img(os.path.join(category, image_name))
#             # Convert the image to numpy array
#             image = img_to_array(image)
#             # Normalize the image
#             image = image.astype('float32') / 255.0
#             # Append the image and its corresponding label to the data and labels list
#             data.append(image)
#             labels.append(to_categorical(category_id, num_classes=len(categories)))

#     return np.array(data), np.array(labels)

# def save_data(data, filename):
#     # Save the preprocessed data
#     np.save(filename, data)

# def load_data(filename):
#     # Load the preprocessed data
#     return np.load(filename)

# # Define categories and number of samples
# categories = ['Data/Mild Dementia', 'Data/Moderate Dementia', 'Data/Non Demented', 'Data/Very mild Dementia']
# num_samples = 488

# # Load and preprocess images
# data, labels = load_and_preprocess_images(categories, num_samples)

# # Split the data into training and testing sets
# train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# # Save data
# save_data(train_data, 'Data/train_data.npy')
# save_data(train_labels, 'Data/train_labels.npy')
# save_data(test_data, 'Data/test_data.npy')
# save_data(test_labels, 'Data/test_labels.npy')

import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt

# Mappings for non-numeric data
hand_mapping = { 'R': 0, 'L': 1 }
gender_mapping = { 'M': 0, 'F': 1 }
label_mapping = {
    'Non Demented': 0,
    'Very Mild Dementia': 1,
    'Mild Dementia': 2,
    'Moderate Dementia': 3,
}

# Load CSV
metadata_path = '../data/raw/demographic_data.csv'
metadata_df = pd.read_csv(metadata_path)
metadata_df = metadata_df.drop(columns=['Delay', 'CDR', ])
metadata_df = metadata_df.dropna()
metadata_df['Hand'] = metadata_df['Hand'].map(hand_mapping)
metadata_df['M/F'] = metadata_df['M/F'].map(gender_mapping)
# Create dictionary for metadata
metadata_df = metadata_df.set_index('ID')
metadata_dict = metadata_df.to_dict(orient='index')

def fetch_metadata(image_id):
    data = metadata_dict.get(image_id.numpy().decode('utf-8'), None)
    if data is not None:
        out = [float(data['M/F']), float(data['Hand']), float(data['Age']), 
               float(data['Educ']), float(data['SES']), float(data['MMSE']), 
               float(data['eTIV']), float(data['nWBV']), float(data['ASF'])]
        valid = [1.0]
    else:
        out = [0.0] * 9
        valid = [0.0]
    return out + valid

def fetch_label_index(label):
    label_index = label_mapping.get(label.numpy().decode('utf-8'), None)
    if label_index is not None:
        out = [label_index]
        valid = [1]
    else:
        out = [0]
        valid = [0]
    return out + valid

def load_and_preprocess_image(image_path):
    valid = True
    # Get ID from image path
    path_parts = tf.strings.split(image_path, os.path.sep)
    filename = path_parts[-1]
    image_id = tf.strings.split(filename, '_mpr')[0]
    # Get metadata for image
    output = tf.py_function(fetch_metadata, [image_id], [tf.float32] * 10)
    metadata = output[:-1]
    metadata_valid = output[-1]
    if tf.equal(metadata_valid, tf.constant([0.0])):
        valid = False
    metadata = tf.stack(metadata)
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image, tf.float32) / 255.0
    # Get label
    label = path_parts[-2]
    output = tf.py_function(fetch_label_index, [label], [tf.int32] * 2)
    label_index = output[:-1]
    label_valid = output[-1]
    if tf.equal(label_valid, tf.constant([0])):
        valid = False
    label = tf.one_hot(label_index[0], depth=len(label_mapping))
    # Return (image, metadata), label
    if valid:
        return (image, tf.concat(metadata, axis=0), image_path), label
    else:
        return (image, tf.concat(metadata, axis=0), image_path), tf.one_hot(-1, depth=len(label_mapping))

def make_image_dataset(data_dir):
    root_data_dir = data_dir + '/*/*.jpg'
    image_path_dataset = tf.data.Dataset.list_files(root_data_dir)
    image_dataset = image_path_dataset.map(load_and_preprocess_image)
    shuffled_image_dataset = image_dataset.shuffle(buffer_size=1000)
    def filter_dataset(data, label):
        return not tf.reduce_all(tf.equal(label, [0, 0, 0, 0]))
    filtered_image_dataset = shuffled_image_dataset.filter(filter_dataset)
    # inspect_image_dataset(filtered_image_dataset, 10)
    return filtered_image_dataset

def make_dataset(data_dir, batch_size):
    image_dataset = make_image_dataset(data_dir)
    dataset = image_dataset.batch(batch_size)
    return dataset

# Debugging
def inspect_image_dataset(dataset, num_samples=3):
    count = 0
    for (image, metadata, image_path), label in dataset.take(num_samples):
        count += 1
        print('Image shape:', image.shape)
        print('Metadata shape:', metadata.shape)
        print('Image path:', image_path)
        print('Label:', label)
        print('Metadata:', metadata)
        # print('Image:')
        # full_image = image * 255
        # plt.imshow(full_image, cmap='gray')
        # plt.show()
    print('Total samples:', count)

# inspect_image_dataset(make_dataset('../data/raw', 128), 100000)