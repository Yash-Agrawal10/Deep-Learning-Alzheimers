# Functions to examine elements of a dataset

def inspect_elements(dataset, num_samples=3):
    print('Inspecting dataset elements...')
    for (image, metadata), label in dataset.take(num_samples):
        print('Image shape:', image.shape)
        print('Metadata shape:', metadata.shape)
        print('Metadata:', metadata)
        print('Label:', label)

from collections import defaultdict
def inspect_size(dataset):
    print('Inspecting dataset size...')
    counts = defaultdict(int)
    for _, label in dataset:
        counts[label.numpy().argmax()] += 1
    total_count = 0
    for label, count in counts.items():
        total_count += count
        print('Label:', label, 'Count:', count)
    print('Total count:', total_count)