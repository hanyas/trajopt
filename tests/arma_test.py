import numpy as np
import numpy.random as npr
from trajopt import core

npr.seed(1337)

if __name__ == '__main__':

    Q = np.eye(2)
    q = np.zeros((2, ))
    q0 = 0.0

    mu = np.zeros((2, ))
    sigma = np.eye(2)

    # expectation of quadratic under gaussian
    print(core.quad_expectation(mu, sigma, Q, q, q0))
