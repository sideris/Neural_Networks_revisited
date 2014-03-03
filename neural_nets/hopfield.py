__author__ = 'Peter George Sideris'
##
#If we want to store new patterns we have two options
#1. Local:
#           A learning rule is local if each weight is updated using information available to neurons on either side
#           of the connection that is associated with that particular weight.
#2. Incremental:
#           New patterns can be learned without using information from the old patterns that have been also used for
#           training. That is, when a new pattern is used for training, the new values for the weights only depend on
#           the old values and on the new pattern
#
# A learning system that would not be incremental would generally be trained only once,
# with a huge batch of training data
##
import random
import numpy as np
from time import sleep
import math


def sign(x):
    return 1 if x >= 0 else -1


class Hopfield:
    def __init__(self, number_neurons):
        self.neurons = number_neurons
        self.create()

    def __init__(self, number_neurons, patterns):
        self.neurons = number_neurons
        self.set_patterns(patterns)
        self.create()

    def init_weights(self):
        self.weights = [[-1.0 + (2.0 * random.random()) for i in xrange(self.neurons)] for i in xrange(self.neurons)]
        for i in range(self.neurons):
            self.weights[i][i] = 0.0
        return self.weights

    def init_states(self):
        self.states = [random.choice([-1, 1]) for i in xrange(self.neurons)]
        return self.states

    def create(self):
        self.init_states()
        self.init_weights()

    def calculate_state(self, old_state):
        rand_bit = int(random.random()*self.neurons)
        old_state[rand_bit] = sign(sum(np.multiply(self.weights[rand_bit], old_state)))
        #self.states = old_state
        return old_state

    def stochastic_state(self, old_state, beta):
        rand_bit = int(random.random()*self.neurons)
        summ = sum(np.multiply(self.weights[rand_bit], old_state))
        g = 1.0/(1.0 + math.exp(-2.0*summ*beta))
        old_state[rand_bit] = 1 if random.random() <= g else -1
        #self.states = old_state
        return old_state

    def set_patterns(self, patt):
        self.patterns = patt

    def train(self):
        n = self.neurons
        if n != len(self.patterns):
            print "Sizes don't match"
            sleep(2)
            print "...Bitch"
            return

        for i in range(n):
            for j in range(n):
                for k in range(len(self.patterns)):
                    self.weights[i][j] += (1.0/n)*self.patterns[k][i]*self.patterns[k][j]
                    if i == j:
                        self.weights[i][j] = 0.0

    def calculate_energy(self):
        self.energy = 0
        for i in range(self.neurons):
            for j in range(self.neurons):
                self.energy += -0.5*self.weights[i][j]*self.states[i]*self.states[j]
        return self.energy