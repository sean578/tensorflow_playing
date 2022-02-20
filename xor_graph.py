"""
Create the forward pass required to classify xor data in low-level tensorflow.
Create the computational graph and display it.
Just input the weights & biases by hand - no optimisation here.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


@tf.function
def forward_pass(inputs, w1, w2, b1, b2, activation_fn):
    """
    Structure:

    x - node
            \
             node - z
            /
    y - node

    Sizes:
    inputs: (n, 2), n = batch size
    w1: (2, 2)
    b1: (1, 2)
    w2: (2, 1)
    b2: (1, 1)
    """

    print('In function')

    linear_1 = inputs @ w1 + b1  # (n, 2)
    activation_1 = activation_fn(linear_1)  # (n, 2)
    linear_2 = activation_1 @ w2 + b2  # (n, 1)
    return activation_fn(linear_2)


if __name__ == '__main__':

    # Create x1, x2 values
    start, stop, num = -2.0, 2.0, 100
    l = np.linspace(start, stop, num)
    data = tf.constant(list(product(l, l)), dtype=tf.float32)

    # Initialise weights and biases
    w1 = tf.constant([[1.0, -1.0], [1.0, -1.0]])
    w2 = tf.constant([[1.0], [1.0]])
    b1 = tf.constant([[-0.5, 1.5]])
    b2 = tf.constant([[-1.5]])

    # Do the forward pass
    # y = forward_pass(data, w1, w2, b1, b2, tf.keras.activations.tanh)

    # Create the graph, pass in the arguments to get the correct signature
    the_graph = forward_pass.get_concrete_function(data, w1, w2, b1, b2, tf.keras.activations.tanh).graph.as_graph_def()
    print('The created graph:')
    print(the_graph)

    # The available graphs (with the allowed signature)
    print('\nAvailable signatures:')
    print(forward_pass.pretty_printed_concrete_signatures())

    # Run the graph
    y = forward_pass(data, w1, w2, b1, b2, tf.keras.activations.tanh)

    # Plot the classifications
    plt.scatter(data[:, 0], data[:, 1], c=y, s=0.5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.show()
