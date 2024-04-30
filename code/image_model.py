import tensorflow as tf
import os
from preprocess import load_dataset, prepare_dataset

# Define constants
input_shape = (248, 496, 1)
num_classes = 4

# Load the dataset
print('Loading dataset...')
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '../data/processed')
batch_size = 64
raw_dataset = load_dataset(data_path)
metadata_dataset = raw_dataset.map(lambda x, y: (x[0], y))
train, validate, test = prepare_dataset(metadata_dataset, batch_size)

# Define the model
print('Defining model...')
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
print('Compiling model...')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print('Training model...')
epochs = 20
history = model.fit(train, epochs=epochs, validation_data=validate)

# Evaluate the model
print('Evaluating model...')
test_loss, test_acc = model.evaluate(test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Save the model
print('Saving model...')
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, '../models/image_model.keras')
model.save(model_path)