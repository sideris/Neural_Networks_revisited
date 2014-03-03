__author__ = 'Peter George Sideris'
from neural_nets import bp
import random as rnd
import numpy as np

def create_random_patterns(number_patterns, bits):
    return [[rnd.choice([-1, 1]) for i in xrange(bits)] for i in xrange(number_patterns)]

back = bp.BackPropagation('train1.txt', 'valid1.txt', 3, 1, 0.02, 0.0001)
back.train()
print back.tobe_permuted
print np.random.permutation(back.tobe_permuted).tolist()
