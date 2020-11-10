"""
smoothlife, implemented from Stephan Rafler's paper.
https://arxiv.org/pdf/1111.1567.pdf
"""

import math
import numpy as np

# birth and death bounds
b_1 = 0
b_2 = 1
d_1 = 0
d_2 = 1


def sigmoid_about_a(x, a, grad=1):

    """sigmoid function of x, with intercept a, and steepness of grad."""
    return 1/(1 + np.exp(-4/grad * (x - a)))

def sigmoid_a_to_b(x, a, b, grad=1):

    """sigmoid function of x between a and b."""
    return sigmoid_about_a(x,a,grad)*(1-sigmoid_about_a(x,b,grad))

def sigmoid_m(x, y, m):

    """sigmoid for area integrals."""
    return x*(1-sigmoid_about_a(m,0.5)) + y*sigmoid_about_a(m,0.5)

def transition_function(n,m):

    """state transition function for smoothlife."""
    sigmoid_a_to_b(n, sigmoid_m(b_1, d_1, m), sigmoid_m(b_2, d_2, m))
