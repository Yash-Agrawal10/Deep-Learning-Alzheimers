from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
NUM_EPOCHS = 12

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.num_classes = 4
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = .001)
        # TODO: Initialize all trainable parameters
        CNN1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1))
        CBias1 = tf.Variable(tf.zeros(16))
        CNN2 = tf.Variable(tf.random.truncated_normal([8,8,16,20], stddev=0.1))
        CBias2 = tf.Variable(tf.zeros(20))
        CNN3 = tf.Variable(tf.random.truncated_normal([8,8,20,20], stddev=0.1))
        CBias3 = tf.Variable(tf.zeros(20))
        CNN4 = tf.Variable(tf.random.truncated_normal([4,4,20,80], stddev=0.1))
        CBias3 = tf.Variable(tf.zeros(20))
        self.epochs = 12
        self.batch_size = 64
        #bias3 = tf.Variable(tf.zeros([2]))
        self.trainable_vars = [CNN1, CBias1, CNN2, CBias2, CNN3, CBias3]
    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        c1_out = tf.nn.conv2d(inputs, self.trainable_vars[0], strides=2, padding='VALID')
        c1_out = tf.nn.bias_add(c1_out, self.trainable_vars[1])
        #c1_out = tf.nn.batch_normalization(c1_out)
        c1_out = tf.nn.relu(c1_out)
        c1_out = tf.nn.max_pool(c1_out, ksize = 3, strides = 2, padding = "VALID")
        c2_out = tf.nn.conv2d(c1_out, self.trainable_vars[2], strides=2, padding='VALID')
        c2_out = tf.nn.bias_add(c2_out, self.trainable_vars[3])
        #c2_out = tf.nn.batch_normalization(c2_out)
        c2_out = tf.nn.relu(c2_out)
        c2_out = tf.nn.max_pool(c2_out, ksize = 2, strides = 2, padding = "VALID")
        c3_out = tf.nn.conv2d(c2_out, self.trainable_vars[4], strides=1, padding='VALID')
        c3_out = tf.nn.bias_add(c3_out, self.trainable_vars[5])
        #c3_out = tf.nn.batch_normalization(c3_out)
        c3_out = tf.nn.relu(c3_out)
        #print(c3_out.shape)
        c4_out = tf.nn.conv2d(c3_out, self.trainable_vars[6], strides = 1, padding = 'VALID')
        dense_in = tf.reshape(c4_out, [inputs.shape[0], 1680])
        dense1_out = tf.keras.layers.Dense(512, activation="relu")(dense_in)
        dense1_out = tf.nn.dropout(dense1_out, rate = 0.1)
        dense2_out = tf.keras.layers.Dense(256, activation = "relu")(dense1_out)
        dense2_out = tf.nn.dropout(dense2_out, rate = 0.1)
        out = tf.keras.layers.Dense(64)(dense2_out)
        #print(out)
        return tf.keras.layers.Dense(4, activation= "softmax")(out)


class MultiModalModel(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.num_classes = 4
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = .001)
        # TODO: Initialize all trainable parameters
        CNN1 = tf.Variable(tf.random.truncated_normal([5,5,180,1800], stddev=0.1))
        CBias1 = tf.Variable(tf.zeros(16))
        CNN2 = tf.Variable(tf.random.truncated_normal([8,8,1800,24000], stddev=0.1))
        CBias2 = tf.Variable(tf.zeros(20))
        CNN3 = tf.Variable(tf.random.truncated_normal([8,8,2400,2400], stddev=0.1))
        CBias3 = tf.Variable(tf.zeros(20))
        CNN4 = tf.Variable(tf.random.truncated_normal([4,4,2400,3600], stddev=0.1))
        CBias3 = tf.Variable(tf.zeros(20))
        self.epochs = 12
        self.batch_size = 64
        #bias3 = tf.Variable(tf.zeros([2]))
        self.trainable_vars = [CNN1, CBias1, CNN2, CBias2, CNN3, CBias3, CNN4]
    def call(self, inputs, demographics, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        c1_out = tf.keras.layers.Conv3D(filters=32, kernel_size=(2, 2, 2), activation='relu')(inputs)
        c2_in = tf.keras.layers.MaxPool3D(pool_size= (4,2,2))(c1_out)
        c2_in = tf.keras.layers.SpatialDropout3D(rate = .1)(c2_in)
        c2_out = tf.keras.layers.Conv3D(filters=12, kernel_size=(4, 4, 4), activation='relu')(c2_in)
        c3_in = tf.keras.layers.MaxPool3D(pool_size= (4,4,8))(c2_out)
        c3_in = tf.keras.layers.SpatialDropout3D(rate = .1)(c3_in)
        c3_out = tf.keras.layers.Conv3D(filters=8, kernel_size=(4, 4, 4), activation='relu')(c3_in)
        c4_in = tf.keras.layers.MaxPool3D(pool_size= (3,6,3))(c3_out)
        c4_out = tf.keras.layers.Conv3D(filters=16, kernel_size=( 3,3,3), activation='relu')(c4_in)
        c4_out = tf.keras.layers.MaxPool3D(pool_size= (2,2,2))(c4_out)

        dense_in = tf.reshape(c4_out, [inputs.shape[0], 256])
        dense_in = tf.concat([dense_in, demographics], axis = 1)
        dense0_out = tf.keras.layers.Dense(512, activation="LeakyRelu")(dense_in)
        dense1_out = tf.keras.layers.Dense(256, activation="relu")(dense0_out)
        dense1_out = tf.nn.dropout(dense1_out, rate = 0.1)
        dense2_out = tf.keras.layers.Dense(96, activation = "relu")(dense1_out)
        dense2_out = tf.nn.dropout(dense2_out, rate = 0.1)
        out = tf.keras.layers.Dense(32)(dense2_out)
        return tf.keras.layers.Dense(4, activation= "softmax")(out)
    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        ce = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(ce)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''
    batches = round(len(train_inputs) / model.batch_size)
    acc = 0
    for i in range(batches):
        indicies = tf.random.shuffle(tf.range(start= i*model.batch_size, limit= (i+1)*model.batch_size, dtype=tf.int32))
        shuff = tf.gather(train_inputs, indicies)
        shufflab = tf.gather(train_labels, indicies)
        shuff = tf.image.random_flip_left_right(shuff)
        with tf.GradientTape() as tape:
            pred = model(shuff)
            loss = model.loss(pred, shufflab)
        grads = tape.gradient(loss, model.trainable_variables)
        #print(grads)
        #print(model.trainable_vars)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc += model.accuracy(pred, shufflab)
    return acc / batches
        


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """

    pred = model(test_inputs, True)
    loss = model.loss(pred, test_labels)
    acc = model.accuracy(pred, test_labels)
    return acc


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    LOCAL_TRAIN_FILE = '../data/train'
    LOCAL_TEST_FILE = '../data/test'

    #train
    m = Model()
    inputs, labels = preprocess(LOCAL_TRAIN_FILE) ###change### ###496x248
    test_inputs, test_labels = preprocess(LOCAL_TEST_FILE) ###change###
    print(len(test_inputs))
    print(len(test_labels))
    for i in range(0, m.epochs):
        print("epoch" + str(i))
        acc = train(m, inputs, labels)
        print("epoch" + str(i) + "accuracy: " + str(acc))
    testing = test(m, test_inputs, test_labels)
    print("testing accuracy: " + str(testing))


if __name__ == '__main__':
    main()
