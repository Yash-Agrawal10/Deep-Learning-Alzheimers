import tensorflow as tf
import os
from preprocess import load_dataset, prepare_dataset

# Load the dataset
print('Loading dataset...')
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '../data/processed')
batch_size = 64
raw_dataset = load_dataset(data_path)
metadata_dataset = raw_dataset.map(lambda x, y: (x[1], y))
train, validate, test = prepare_dataset(metadata_dataset, batch_size)

# Define the model
metadata_shape = (9,)
num_classes = 4
print('Defining model...')
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
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
epochs = 5
history = model.fit(train, epochs=epochs, validation_data=validate)

# Evaluate the model
print('Evaluating model...')
test_loss, test_acc = model.evaluate(test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Save the model
print('Saving model...')
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, '../models/metadata_model.keras')
model.save(model_path)