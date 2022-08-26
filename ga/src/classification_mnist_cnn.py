from create_log import create_logger
logger = create_logger('mnist-classifier.log')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import sys
import time
import random
import numpy as np
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.layers import BatchNormalization

# baseline cnn model for mnist
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
 

from sklearn.metrics import accuracy_score, f1_score

import ga

logger.info('')
logger.info('=========================================')
logger.info('Starting classification_mnist.py')
logger.info('')

# Set as many random states as possible
random_state = 42
random.seed (random_state)
np.random.seed(random_state)
os.environ['PYTHONHASHSEED'] = str (random_state)

# Definition of global constants
N_FEATURES = 28 * 28
MAX_LAYERS = 6


def classify_mnist(data: ga.GaData, crit: ga.Criterion):
    cpu = time.time()
    
    # fetch the ML data
    X_train = data.X_train
    X_val = data.X_val
    y_train = data.y_train
    y_val = data.y_val
    
    # get the NN configaration data
    #layer_sizes = data.data_dict['layer_sizes']
    #kernel_sizes = data.data_dict['kernel_sizes']
    verbose = data.data_dict['verbose']
    epochs = data.data_dict['n_epochs']
    batch_size = data.data_dict['batch_size']
    layers = data.data_dict['layers']
    kernels = data.data_dict['kernels']

    if len(layers) != len(kernels):
        raise ValueError(f'Size of kernels ({len(kernel_sizes)}) and layers ({len(layer_sizes)}) '
                         f'are not equal')
    model = Sequential()

    input_nodes = data.X_train.shape[1]
    output_nodes = data.y_train.shape[1]
    
    for i in range(len(layers)):
        layer_size = layers[i]
        kernel_size = kernels[i]
        #logger.info(str(layers) + ' --- ' + str(kernels))
        #logger.info(f'Layer: {i}: layer_size {layer_size}, kernel_size {kernel_size}')

        try:
            if i == 0:
                model.add(Conv2D(layer_size, (kernel_size, kernel_size), activation='relu', 
                                 kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
                                 #input_dim=input_nodes))
            else:
                model.add(Conv2D(layer_size, (kernel_size, kernel_size), activation='relu', 
                                 kernel_initializer='he_uniform'))

            # if

            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))

            #logger.info(f'Created: {i}: layer_size {layer_size}, kernel_size {kernel_size}')

        # due to the nature of creating layersizes and kernel sizes quite some settings
        # lead to an impossible configuration. In that case a crash occurs and 
        # generating new layers should be stopped.
        # Caveat: when an exception occurs for some other reason it will be ignored, just
        # output the description of the exception
        except Exception as e:
            logger.debug(f'Error -> Layer: {i}: layer_size {layer_size}, kernel_size {kernel_size}')
            logger.debug(f'> Exception (ignored): {str(e)}')
            logger.debug(data)

            break

    # for

    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(output_nodes, activation='softmax'))

    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    #model.add(Dense(output_nodes, kernel_initializer='glorot_normal', activation='softmax'))

    #adam = Adam(learning_rate=1e-3)
    #model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    if verbose >= 1:
        model.summary()

    hist = model.fit(X_train, y_train,   
                     validation_data=(X_val, y_val), 
                     #val_generator, 
                     #validation_steps=10,
                     epochs=epochs,
                     #steps_per_epoch=100,
                     batch_size=batch_size,
                     #callbacks=[checkpoint, early],
                     verbose=verbose)

    #best_model = load_model('vgg16_1.h5')
    y_pred = model.predict(X_val) # .astype(y_val.dtype)
    
    val_label = np.argmax(y_val, axis=1) # act_label = 1 (index)
    pred_label = np.argmax(y_pred, axis=1) # pred_label = 1 (index)

    cpu = time.time() - cpu
    
    val_acc = accuracy_score(val_label, pred_label, normalize=True)
    val_f1 = f1_score(val_label, pred_label, average='weighted')
    acc_cpu = val_acc / cpu

    return [val_acc, val_f1, acc_cpu]

### classify_mnist ###


def read_mnist():
    from tensorflow.keras.datasets import mnist

    # load mnist data set
    (X_train, y_train), (X_val, y_val) = mnist.load_data()

    # reshape X to fit for CNN
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
    y_val = y_val.reshape(y_val.shape[0], 1)

    # normalize X
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    # and one hot encode y
    y_train = OneHotEncoder(sparse=False).fit_transform (y_train).astype(np.int32)
    y_val = OneHotEncoder(sparse=False).fit_transform (y_val).astype(np.int32)

    logger.info('X_train.shape = ' + str(X_train.shape) + 'type = ' + str(X_train.dtype))
    logger.info('y_train.shape = ' + str(y_train.shape) + 'type = ' + str(y_train.dtype))
    logger.info('X_val.shape =   ' + str(X_val.shape) + 'type = ' + str(X_val.dtype))
    logger.info('y_val.shape =   ' + str(y_val.shape) + 'type = ' + str(y_val.dtype))

    return X_train, y_train, X_val, y_val

### read_data ###


def plot(X):
    # Plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)

        # plot raw pixel data
        plt.imshow(X[i], cmap=plt.get_cmap('gray'))

    # for

    # show the figure
    plt.show()

### plot ###


X_train, y_train, X_val, y_val = read_mnist()
plot(X_train)

# set parameter for neural network and ga
layer_sizes: list = [32, 64, 128, 256, 512, 1024]
kernel_sizes: list = [3, 5, 7]
pop_size: int = 5
n_epochs: int = 10
n_layers: int = 3
max_layer_size: int = len(layer_sizes)
batch_size: int = 128

fitnesses = ['val_acc', 'val_f1', 'val/cpu']
pop = ga.Population(p_mutation=0.25, p_crossover=5, 
                    fitness=fitnesses, selection_key=fitnesses[1])

data = ga.GaData(X_train, X_val, None, y_train, y_val, None)

data.register_variable('verbose', 1)
data.register_variable('n_epochs', n_epochs)
data.register_variable('batch_size', batch_size)
data.register_variable('n_layers', n_layers)
data.register_variable('layers', [32])
data.register_variable('kernels', [3])

crit = ga.Criterion('val_f1', 'ge', 1.0)

#pop.add_var_int('n_epochs', 10, 25)
pop.add_var_int('n_layers', 1, 4)
pop.add_var_int_array('layers', 'I', length='n_layers', value_list=layer_sizes)
pop.add_var_int_array('kernels', 'I', length='n_layers', value_list=kernel_sizes)

cpu = time.time()

fitness = classify_mnist(data, crit)
logger.info('validation accuracy: {:.2f}'.format(fitness[0]))
logger.info('validation F1:       {:.2f}'.format(fitness[1]))
logger.info('val accuracy / cpu:  {:.2f}'.format(fitness[2]))

cpu = time.time() - cpu
logger.info(f'CPU is {cpu:.2f}')

data.data_dict['verbose'] = 0
data.data_dict['n_epochs'] = n_epochs

pop.set_fitness_function(classify_mnist, data)
pop.create_population(pop_size)

pop.population[1].show()

logger.info('')
logger.info('--- Initial Generation ---')
pop.pre_compute(crit)
pop.show()

for generation in range(1, 11):
    cpu = time.time()

    logger.info('')
    logger.info('*** Generation ' + str(generation))
    pop.next_generation(pop_size, crit)
    pop.show()

    cpu = time.time() - cpu
    logger.info(f'CPU for this generation: {cpu}')

logger.info('[Ready]')
