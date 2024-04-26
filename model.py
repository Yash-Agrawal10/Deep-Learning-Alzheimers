# Load the preprocessed data
from preprocess_images import load_data

train_data = load_data('Data/data.npy')
train_labels = load_data('Data/labels.npy')

print(train_data.shape)