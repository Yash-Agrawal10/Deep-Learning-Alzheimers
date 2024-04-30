import sys
sys.path.append('../')
from preprocess import get_dataset
from test_dataset import inspect_elements, inspect_size

# Testing
dataset = get_dataset()
inspect_elements(dataset)
inspect_size(dataset)