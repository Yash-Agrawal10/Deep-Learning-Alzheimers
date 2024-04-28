from preprocess_images import make_dataset
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
from tqdm import tqdm

# Load the preprocessed data
data = make_dataset('../data/raw', 128)
num_batches = 256 # hardcoding for now

num_train = int(0.7 * num_batches)
num_test = num_batches - num_train

# Split the data (which is tf dataset) into training and testing sets
train_data = data.take(num_train)
test_data = data.skip(num_train)

test_data = test_data.map(lambda x, y: ((x[0], x[1]), y))

# Separate the inputs and labels in the test data
test_inputs = test_data.map(lambda x, y: x)
test_labels = test_data.map(lambda x, y: y)

# Combine the inputs and labels back into a single dataset
test_data = tf.data.Dataset.zip((test_inputs, test_labels))

# create a convolutional neural network
def create_model():
    # model = tf.keras.Sequential([
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
    
    inputA = Input(shape=(248, 496))
    inputB = Input(shape=(9,)) 
    x = Flatten()(inputA)
    x = Dense(64, activation="relu")(x)

    # Second branch (for metadata)
    y = Dense(10, activation="relu")(inputB)  # replace 10 with the number of nodes you want

    # Combine the output of the two branches
    combined = concatenate([x, y])

    # Apply another dense layer and then the final output layer
    z = Dense(64, activation="relu")(combined)
    z = Dense(4, activation="softmax")(z)

    # Our final model will accept inputs from the two branches and output a single value
    model = Model(inputs=[inputA, inputB], outputs=z)

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
            image, metadata, _ = batch[0]
            label = batch[1]
            metrics = model.train_on_batch([image, metadata], label)
            pbar.set_description(f"loss: {metrics[0]:.4f}, accuracy: {metrics[1]:.4f}")
            pbar.update()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, steps=num_test)
print(f"Test accuracy: {test_acc}")
