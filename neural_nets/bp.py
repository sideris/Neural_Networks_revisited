__author__ = 'PeterGeorge'
#####
# V is the state of the neurons of the hidden layer (array)
#   --threshold V
#     --delta threshold V
#   --weights_to_hidden
#     --Bj
#     --delta_j
#
# O is the output (array)
#   --threshold O
#     --delta threshold O
#   --weights_to_out
#     --Bi
#     --delta_i
#
# The training and validation sets should be:
# inputs: betweens -1 and 1 separated by ","
# expected output: last element in each line
# otherwise overwrite the set_sets
#####
import random

def sign(x):
    return 1 if x >= 0 else -1


class BackPropagation:

    def __init__(self, training, validation, number_hidden, number_out, beta, eta):
        self.__set_sets(training, validation)

        self.inputs = len(self.training[:][0]) - number_out
        self.hidden = number_hidden
        self.outputs = number_out

        self.beta = beta
        self.eta = eta
        self.H = 1000

        self.weights_to_hidden = [[-1.0 + (2.0 * random.random()) for i in xrange(self.inputs)] for i in xrange(number_hidden)]

        self.V = [0.0 for i in xrange(number_hidden)]
        self.threshold_V = [-1.0 + (2.0 * random.random()) for i in xrange(number_hidden)]
        self.delta_threshhold_V = [0.0 for i in xrange(number_hidden)]
        self.Bj = [0.0 for i in xrange(number_hidden)]
        self.delta_j = [0.0 for i in xrange(number_hidden)]

        self.weights_to_out = [[-1.0 + (2.0 * random.random()) for i in xrange(number_hidden)] for i in xrange(number_out)]

        self.O = [0.0 for i in xrange(number_out)]
        self.threshold_O = [-1.0 + (2.0 * random.random()) for i in xrange(number_out)]
        self.delta_threshold_O = [0.0 for i in xrange(number_out)]
        self.Bi = [0.0 for i in xrange(number_out)]
        self.delta_i = [0.0 for i in xrange(number_out)]

    def __set_sets(self, training, validation):
        #count all the lines for initilization
        file_train = open(training, 'r')
        file_val = open(validation, 'r')
        train_len = 0
        for line in file_train:
            if train_len == 0:
              train_len2 = len([x for x in line.split(',')])
            train_len += 1
        val_len = 0
        for line in file_val:
            val_len += 1

        file_train.close()
        file_val.close()

        self.training = [[0.0 for i in xrange(train_len2)] for i in xrange(train_len)]
        self.validation = [[0.0 for i in xrange(train_len2)] for i in xrange(val_len)]

        #read the actual file and save data
        file_train = open(training, 'r')
        file_val = open(validation, 'r')

        count = 0
        for line in file_train:
            self.training[count][:] = [float(x.strip()) for x in line.split(',')]
            count += 1
        count = 0
        for line in file_val:
            self.validation[count][:] = [float(x.strip()) for x in line.split(',')]
            count += 1

        file_train.close()
        file_val.close()

    def train(self):

        tobe_permuted = [i for i in xrange(len(self.training[:]))]
        while self.H >= 1:
            random.shuffle(tobe_permuted)
