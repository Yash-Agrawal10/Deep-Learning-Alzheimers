import os
import sys
sys.path.append('../')
from preprocess import load_dataset
from test_dataset import inspect_elements, inspect_size

# Load saved data
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../../data')
dataset_path = os.path.join(data_dir, 'processed')
if not os.path.exists(dataset_path):
    raise Exception('Dataset not found')
dataset = load_dataset(dataset_path)

# Testing
inspect_elements(dataset)
inspect_size(dataset)