"""
Put the forward pass of xor into a module - subclassing tf.Module
To see tensorboard display of graph run in root of repo:
tensorboard --logdir=logs
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime


class XorModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        # Initialise weights and biases
        self.w1 = tf.Variable([[1.0, -1.0], [1.0, -1.0]])
        self.w2 = tf.Variable([[1.0], [1.0]])
        self.b1 = tf.Variable([[-0.5, 1.5]])
        self.b2 = tf.Variable([[-1.5]])

    @tf.function
    def __call__(self, x, activation_fn):
        linear_1 = x @ self.w1 + self.b1  # (n, 2)
        activation_1 = activation_fn(linear_1)  # (n, 2)
        linear_2 = activation_1 @ self.w2 + self.b2  # (n, 1)
        return activation_fn(linear_2)


def tensorboard_logging_setup():
    # Set up logging.
    # Return the writer & log directory

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/func/%s" % stamp
    return tf.summary.create_file_writer(logdir), logdir


if __name__ == '__main__':

    writer, logdir = tensorboard_logging_setup()

    # Create x1, x2 values
    start, stop, num = -2.0, 2.0, 100
    l = np.linspace(start, stop, num)
    data = tf.constant(list(product(l, l)), dtype=tf.float32)

    # Create the module
    xor_module = XorModule(name='xor')
    # Run the forward pass
    tf.summary.trace_on(graph=True)
    tf.profiler.experimental.start(logdir)

    y = xor_module(data, tf.keras.activations.tanh)

    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)

    # See some info about the module
    print("\ntrainable variables:\n")
    for var in xor_module.trainable_variables:
        print(var)

    # Plot the classifications
    plt.scatter(data[:, 0], data[:, 1], c=y, s=0.5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.show()
