import os
import sys
sys.path.append('../')
from preprocess import load_dataset
from test_dataset import inspect_elements, inspect_size, inspect_groups

# Load saved data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../../data/processed')

train_path = os.path.join(data_dir, 'train.tfrecord')
if not os.path.exists(train_path):
    raise Exception('Dataset not found')
train = load_dataset(train_path)

test_path = os.path.join(data_dir, 'test.tfrecord')
if not os.path.exists(test_path):
    raise Exception('Dataset not found')
test = load_dataset(test_path)

# Testing
print('Inspecting train dataset...')
inspect_elements(train)
inspect_size(train)
inspect_groups(train)

print('Inspecting test dataset...')
inspect_elements(test)
inspect_size(test)
inspect_groups(test)