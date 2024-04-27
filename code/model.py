from preprocess_images import make_dataset
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Load the preprocessed data
data = make_dataset('../data/raw', 128)
num_batches = 256 # hardcoding for now

num_train = int(0.7 * num_batches)
num_test = num_batches - num_train

# Split the data (which is tf dataset) into training and testing sets and extract labels
train_data = data.take(num_train)
test_data = data.skip(num_train)

# create a convolutional neural network
def create_model():
    model = tf.keras.Sequential([
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Dropout(0.25),
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Dropout(0.25),
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
model = create_model()


for epoch in range(1):
    print(f"Epoch {epoch+1}")
    with tqdm(total=num_train, dynamic_ncols=True) as pbar:
        for batch in train_data:
            image, _, _ = batch[0]
            label = batch[1]
            metrics = model.train_on_batch(image, label)
            pbar.set_description(f"loss: {metrics[0]:.4f}, accuracy: {metrics[1]:.4f}")
            pbar.update()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, steps=num_test)
print(f"Test accuracy: {test_acc}")
