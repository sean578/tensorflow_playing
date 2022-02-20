import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def step(x):
    """ Return element wise 1 if >= 0, -1 if < 0 """

    return 2*np.heaviside(x, 0) - 1


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

    linear_1 = inputs @ w1 + b1  # (n, 2)
    activation_1 = activation_fn(linear_1)  # (n, 2)
    linear_2 = activation_1 @ w2 + b2  # (n, 1)
    return activation_fn(linear_2)


if __name__ == '__main__':

    # Create x1, x2 values
    start, stop, num = -2, 2, 100
    l = np.linspace(start, stop, num)
    data = np.array(list(product(l, l)))

    # Initialise weights and biases
    w1 = np.array([[1, -1], [1, -1]])
    w2 = np.array([[1], [1]])
    b1 = np.array([[-0.5, 1.5]])
    b2 = np.array([[-1.5]])

    # Do the forward pass
    y = forward_pass(data, w1, w2, b1, b2, step)

    # Plot the classifications
    plt.scatter(data[:, 0], data[:, 1], c=y, s=0.5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.show()
