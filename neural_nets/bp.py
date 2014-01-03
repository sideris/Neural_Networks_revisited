__author__ = 'PeterGeorge'

import random
import numpy as np
from time import sleep
import math


def sign(x):
    return 1 if x >= 0 else -1


class BackPropagation:

    def __init__(self, number_inputs, number_hidden, number_out):
        self.weights_to_hidden = [[-1.0 + (2.0 * random.random()) for i in xrange(number_inputs)] for i in xrange(number_hidden)]
        self.weights_to_out = [[-1.0 + (2.0 * random.random()) for i in xrange(number_hidden)] for i in xrange(number_out)]
        self.threshold_hidden = [-1.0 + (2.0 * random.random()) for i in xrange(number_hidden)]
        self.threshold_out = [-1.0 + (2.0 * random.random()) for i in xrange(number_out)]
        self.V = [0.0 for i in xrange(number_hidden)]
        self.O = [0.0 for i in xrange(number_out)]