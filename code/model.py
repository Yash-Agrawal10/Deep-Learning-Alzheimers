from preprocess_images import make_dataset
import tensorflow as tf
import numpy as np

# Load the preprocessed data
data = make_dataset('../data/raw', 32)

# Split the data (which is tf dataset) into training and testing sets and extract labels
train_size = int(0.7 * len(list(data)))  # 70% of data for training
test_size = int(0.3 * len(list(data)))   # 30% of data for testing

train_data = data.take(train_size)
test_data = data.skip(train_size)

# create a convolutional neural network
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(248, 256, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
model = create_model()
model.fit(train_data, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)
