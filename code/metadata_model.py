import tensorflow as tf
import os
from preprocess import load_dataset, prepare_dataset

# Load the dataset
print('Loading dataset...')
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data/processed')
batch_size = 64

train_path = os.path.join(data_dir, 'train.tfrecord')
train_dataset = load_dataset(train_path)
train_dataset = train_dataset.map(lambda x, y: (x[1], y))
train_dataset = prepare_dataset(train_dataset, batch_size)

test_val_path = os.path.join(data_dir, 'test.tfrecord')
test_val_dataset = load_dataset(test_val_path)
test_val_dataset = test_val_dataset.map(lambda x, y: (x[1], y))
test_val_length = sum(1 for _ in test_val_dataset)
test_length = int(test_val_length * 0.66)

test_dataset = test_val_dataset.take(test_length)
test_dataset = prepare_dataset(test_dataset, batch_size)

val_dataset = test_val_dataset.skip(test_length)
val_dataset = prepare_dataset(val_dataset, batch_size)

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
epochs = 20
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

# Evaluate the model
print('Evaluating model...')
test_loss, test_acc = model.evaluate(test_dataset)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Save the model
print('Saving model...')
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, '../models/metadata_model.keras')
model.save(model_path)