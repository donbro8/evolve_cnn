from block_cnn import BlockNEAT
import pprint
import numpy as np
from copy import deepcopy
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.utils import plot_model, to_categorical

my_class = BlockNEAT('/Users/Donovan/Documents/Masters/masters-ed02/gblock/parameters.yaml')

(X_train, Y_train), (X_test, Y_test) = my_class.load_data(use_case = 'mnist')

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

X_train, X_test, Y_train, Y_test = my_class.get_validation_data(X_train, Y_train, validation_split = 0.75)
X_train, X_test, Y_train, Y_test = my_class.get_validation_data(X_train, Y_train, validation_split = 0.15)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# convert class vectors to binary class matrices
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# The population is entering a loop where there are 10 unique species with only one individual present, which means no offspring is being generated

my_class.evolve_block(X_test, Y_test)