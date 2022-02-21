"""
Put the forward pass of xor into a module - subclassing tf.keras.Model
Train with custom loss and training loop
To see tensorboard display of graph run in root of repo:
tensorboard --logdir=logs
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime


class XorModule(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)

        # Initialise weights and biases
        # todo: use a keras initialiser?
        self.w1 = tf.Variable([[0.1, 0.3], [-0.1, 0.2]])
        self.w2 = tf.Variable([[0.3], [-0.2]])
        self.b1 = tf.Variable([[-0.5, 0.2]])
        self.b2 = tf.Variable([[0.1]])

    @tf.function
    def call(self, x):
        linear_1 = x @ self.w1 + self.b1  # (n, 2)
        activation_1 = tf.keras.activations.tanh(linear_1)  # (n, 2)
        linear_2 = activation_1 @ self.w2 + self.b2  # (n, 1)
        return tf.keras.activations.sigmoid(linear_2)


def train(model, x, y, learning_rate):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    with tf.GradientTape() as t:
        current_loss = loss(y, model(x))

    dw1, dw2, db1, db2 = t.gradient(current_loss, [model.w1, model.w2, model.b1, model.b2])

    model.w1.assign_sub(learning_rate * dw1)
    model.w2.assign_sub(learning_rate * dw2)
    model.b1.assign_sub(learning_rate * db1)
    model.b2.assign_sub(learning_rate * db2)


def training_loop(model, x, y, learning_rate, epochs):

    for epoch in range(epochs):
        train(model, x, y, learning_rate)
        current_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y, model(x))
        print('Epoch', epoch, 'Loss:', current_loss)


def tensorboard_logging_setup():
    # Set up logging.
    # Return the writer & log directory

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/func/%s" % stamp
    print('Logdir:', logdir)
    return tf.summary.create_file_writer(logdir), logdir


def tensorboard_start_logging(logdir):
    tf.summary.trace_on(graph=True)
    tf.profiler.experimental.start(logdir)


def tensorboard_write(logdir):
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)


def create_data(start, stop, num):
    l = np.linspace(start, stop, num)
    data = tf.constant(list(product(l, l)), dtype=tf.float32)
    y = []
    for x1, x2 in data:
        if 1.5 < x1 + x2 or x1 + x2 < 0.5:
            y.append(0)
        else:
            y.append(1)

    y = np.array(y)
    y = y[:, np.newaxis]
    return data, y


if __name__ == '__main__':
    # User params
    data_start, data_stop, num_points = -2.0, 2.0, 100
    learning_rate = 3.0
    epochs = 500

    # Set up tensorboard
    writer, logdir = tensorboard_logging_setup()

    # Create the data
    data, y = create_data(data_start, data_stop, num_points)

    # Create the module
    xor_module = XorModule(name='xor')

    # Print variables before training
    print("\nBefore training:\n")
    for var in xor_module.trainable_variables:
        print(var)

    # Train the model
    tensorboard_start_logging(logdir)
    training_loop(xor_module, data, y, learning_rate, epochs)
    tensorboard_write(logdir)

    # Print variables after training
    print("\nAfter training:\n")
    for var in xor_module.trainable_variables:
        print(var)

    # Plot the classifications
    plt.scatter(data[:, 0], data[:, 1], c=xor_module(data), s=0.5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.show()
