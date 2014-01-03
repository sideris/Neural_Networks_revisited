__author__ = 'Peter George Sideris'
from neural_nets import hopfield as hp
import random as rnd

def create_random_patterns(number_patterns, bits):
    return [[rnd.choice([-1, 1]) for i in xrange(bits)] for i in xrange(number_patterns)]

neurons = 100

