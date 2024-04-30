import tensorflow as tf
import os
import pandas as pd

# Data directory path
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data')

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
metadata_path = os.path.join(data_dir, 'raw/demographic_data.csv')
metadata_df = pd.read_csv(metadata_path)
metadata_df = metadata_df.drop(columns=['Delay', 'CDR', ])
metadata_df = metadata_df.dropna()
metadata_df['Hand'] = metadata_df['Hand'].map(hand_mapping)
metadata_df['M/F'] = metadata_df['M/F'].map(gender_mapping)
# Create dictionary for metadata
metadata_df = metadata_df.set_index('ID')
metadata_dict = metadata_df.to_dict(orient='index')

##################################
# Code to preprocess the dataset #
##################################

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
    metadata = tf.reshape(metadata, [9, ])
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [248, 496])
    # Get label
    label = path_parts[-2]
    output = tf.py_function(fetch_label_index, [label], [tf.int32] * 2)
    label_index = output[:-1]
    label_valid = output[-1]
    if tf.equal(label_valid, tf.constant([0])):
        valid = False
    label = tf.one_hot(label_index[0], depth=len(label_mapping))
    label.set_shape([len(label_mapping)])
    # Return (image, metadata), label
    if valid:
        return (image, tf.concat(metadata, axis=0)), label
    else:
        return (image, tf.concat(metadata, axis=0)), tf.one_hot(-1, depth=len(label_mapping))

def make_dataset():
    root_data_dir = os.path.join(data_dir, 'raw/*/*.jpg')
    image_path_dataset = tf.data.Dataset.list_files(root_data_dir)
    image_dataset = image_path_dataset.map(load_and_preprocess_image)
    def filter_dataset(data, label):
            return not tf.reduce_all(tf.equal(label, [0, 0, 0, 0]))
    filtered_dataset = image_dataset.filter(filter_dataset)
    return filtered_dataset

def balance_dataset(dataset, num_of_labels, count):
    data_slices = []
    for i in range(num_of_labels):
        target_label = tf.one_hot(i, depth=len(label_mapping))
        def filter_dataset(data, label):
            return tf.reduce_all(tf.equal(label, target_label))
        data_slice = dataset.filter(filter_dataset).take(count)
        data_slices.append(data_slice)
    if not data_slices:
        return None
    balanced_dataset = data_slices[0]
    for data_slice in data_slices[1:]:
        balanced_dataset = balanced_dataset.concatenate(data_slice)
    return balanced_dataset

def get_dataset():
    image_dataset = make_dataset()
    balanced_dataset = balance_dataset(image_dataset, 4, 484)
    return balanced_dataset

#################################
# Code to save/load the dataset #
#################################

def to_bytes_list(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(image, metadata, label):
    feature = {
        'image': to_bytes_list(tf.io.serialize_tensor(image)),
        'metadata': to_bytes_list(tf.io.serialize_tensor(metadata)),
        'label': to_bytes_list(tf.io.serialize_tensor(label))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def save_dataset(dataset, filepath):
    with tf.io.TFRecordWriter(filepath) as writer:
        for (image, metadata), label in dataset:
            example = serialize_example(image, metadata, label)
            writer.write(example)

def parse_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'metadata': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example, feature_description)
    image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)
    image = tf.reshape(image, [248, 496, 1])
    metadata = tf.io.parse_tensor(parsed_features['metadata'], out_type=tf.float32)
    metadata = tf.reshape(metadata, [9, ])
    label = tf.io.parse_tensor(parsed_features['label'], out_type=tf.float32)
    label = tf.reshape(label, [4, ])
    return (image, metadata), label

def load_dataset(filepath):
    raw_dataset = tf.data.TFRecordDataset(filepath)
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    return parsed_dataset

########################################
# Utility for interacting with dataset #
########################################

def prepare_dataset(dataset, batch_size, cache=True, shuffle_buffer_size=1000):
    dataset = dataset.cache() if cache else dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size)

    dataset_size = dataset.reduce(0, lambda x, _: x + 1).numpy()
    train_count = int(0.7 * dataset_size)
    val_count = int(0.1 * dataset_size)
    test_count = dataset_size - train_count - val_count

    train_dataset = dataset.take(train_count).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = dataset.skip(train_count).take(val_count).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = dataset.skip(train_count + val_count).take(test_count).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

############################################
# Code to actually create and save dataset #
############################################

def main():
    print('Creating and saving dataset...')
    dataset = get_dataset()
    save_dataset(dataset, os.path.join(data_dir, 'processed'))
    print('Dataset created and saved!')

if __name__ == '__main__':
    main()