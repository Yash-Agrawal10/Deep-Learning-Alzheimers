import tensorflow as tf
import os
from preprocess import load_dataset, prepare_dataset

# Load the dataset
print('Loading dataset...')
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '../data/processed')
batch_size = 64
raw_dataset = load_dataset(data_path)
multimodal_dataset = raw_dataset
train, validate, test = prepare_dataset(multimodal_dataset, batch_size)

# Define the model
print('Defining model...')
image_shape = (248, 496, 1)
metadata_shape = (9,)
num_classes = 4

image_input = tf.keras.Input(shape=image_shape, name='image_input')
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

metadata_input = tf.keras.Input(shape=metadata_shape, name='metadata_input')
y = tf.keras.layers.Dense(64, activation='relu')(metadata_input)

combined = tf.keras.layers.concatenate([x, y])
z = tf.keras.layers.Dense(64, activation='relu')(combined)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(z)

model = tf.keras.Model(inputs=[image_input, metadata_input], outputs=output)

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
model_path = os.path.join(script_dir, '../models/multimodal_model.keras')
model.save(model_path)