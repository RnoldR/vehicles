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
import math
import random
import warnings
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from keras.datasets.mnist import load_data
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from sklearn.metrics import accuracy_score, f1_score

from ga import ga

logger.info('')
logger.info('=====================================')
logger.info('Starting classification of MNIST data')
logger.info('=====================================')
logger.info('')

# Set as many random states as possible
random_state = 42
random.seed (random_state)
np.random.seed(random_state)
os.environ['PYTHONHASHSEED'] = str (random_state)

# Definition of global constants
N_FEATURES = 28 * 28
MAX_LAYERS = 6


def read_mnist_data():
    data = ga.load_data('mnist')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1:].values

    logger.info('')
    logger.info('X.shape = ' + str(X.shape) + ' type = ' + str(X.dtype))
    logger.info('y.shape = ' + str(y.shape) + ' type = ' + str(y.dtype))
    
    return X, y

### read_mnist_data ###


def prepare_data_mnist_rfc(X, y, split_fraction):
    # reshape y
    y = y.reshape(y.shape[0],)

    # split in training and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      train_size = split_fraction,
                                                      shuffle = False)  

    # Rescale the predictors for better performance
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_val = sc_X.transform(X_val)

    logger.info('')
    logger.info('Data prepared for ML classifier')
    logger.info('X_train.shape = ' + str(X_train.shape) + ' type = ' + str(X_train.dtype))
    logger.info('X_val.  shape = ' + str(X_val.shape) + ' type = ' + str(X_val.dtype))
    logger.info('y_train.shape = ' + str(y_train.shape) + ' type = ' + str(y_train.dtype))
    logger.info('y_val  .shape = ' + str(y_val.shape) + ' type = ' + str(y_val.dtype))

    return X_train, X_val, None, y_train, y_val, None

### prepare_data_mnist_rfc ###    


def prepare_data_mnist_dense(X, y, split_fraction):
    y = y.reshape(y.shape[0], 1)

    # Rescale the predictors for better performance
    X = X / 255

    # and one hot encode y
    y = OneHotEncoder (sparse=False).fit_transform (y).astype(np.int32)

    # reshape y
    #y = y.reshape(y.shape[0],)

    # split in training and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      train_size = split_fraction,
                                                      shuffle = False)  

    logger.info('')
    logger.info('Data prepared for dense neural network')
    logger.info('X_train.shape = ' + str(X_train.shape) + ' type = ' + str(X_train.dtype))
    logger.info('X_val.  shape = ' + str(X_val.shape) + ' type = ' + str(X_val.dtype))
    logger.info('y_train.shape = ' + str(y_train.shape) + ' type = ' + str(y_train.dtype))
    logger.info('y_val  .shape = ' + str(y_val.shape) + ' type = ' + str(y_val.dtype))

    return X_train, X_val, None, y_train, y_val, None

### prepare_data_mnist_rfc ###    


def prepare_data_mnist_cnn(X, y, split_fraction):
    # reshape X to fit for CNN
    X = X.reshape(X.shape[0], 28, 28, 1)
    y = y.reshape(y.shape[0], 1)

    # Rescale the predictors for better performance
    X = X / 255

    # and one hot encode y
    y = OneHotEncoder (sparse=False).fit_transform (y).astype(np.int32)

    # reshape y
    y = y.reshape(y.shape[0],)

    # split in training and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      train_size = split_fraction,
                                                      shuffle = False)  

    logger.info('')
    logger.info('Data prepared for convolutional neural network')
    logger.info('X_train.shape = ' + str(X_train.shape) + ' type = ' + str(X_train.dtype))
    logger.info('X_val.  shape = ' + str(X_val.shape) + ' type = ' + str(X_val.dtype))
    logger.info('y_train.shape = ' + str(y_train.shape) + ' type = ' + str(y_train.dtype))
    logger.info('y_val  .shape = ' + str(y_val.shape) + ' type = ' + str(y_val.dtype))

    return X_train, X_val, None, y_train, y_val, None

### prepare_data_mnist_rfc ###    


def fitness_log_reg(data: ga.Data, criterion: ga.Criterion):
    # fetch the ML data
    X_train = data.X_train
    X_val = data.X_val
    y_train = data.y_train
    y_val = data.y_val

    # fetch the parameters for the logistic regression from data
    C = data.data_dict['C']
    max_iter = data.data_dict['max_iter']
    
    # intercept warnings when they accur and handle accordingly
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Create a logistic regression classifier
        classifier = LogisticRegression(C=C, max_iter=max_iter)#, solver='saga')
        classifier.fit(X_train, y_train)
    
    # if a warning occurs the result is useless, the phenotype
    # is dead on arraival (DoA)
    try:
        warn = w[-1].category
        logger.debug('*** Warning occurred: ', w[-1].category)    

        if warn is sklearn.exceptions.ConvergenceWarning:
            logger.debug('*** Convergence warning')

        logger.debug(w[-1].message)

        logger.info('Dead on Arrival')

        return None

    # Exception, no category = no warning = ok
    except:
        # No warning occurred, the phenotype is usable
        y_pred = classifier.predict(X_val)

        val_acc = accuracy_score(y_val, y_pred, normalize=True)
        val_f1 = f1_score(y_val, y_pred, average='weighted')

        logger.info(f'Ok in {classifier.n_iter_} iterations, val_acc {val_acc:.2f}, val_f1 {val_f1:.2f}')

    # try..except

    return {'val_acc': val_acc,
            'val_f1': val_f1}

### fitness_log_reg ###


def fitness_rfc(data: ga.Data, criterion: ga.Criterion):
    # fetch the ML data
    X_train = data.X_train
    X_val = data.X_val
    y_train = data.y_train
    y_val = data.y_val
    
    # fetch the parameters for the logistic regression from data
    n_estimators = data.data_dict['n_estimators']
    
    # Create a logistic regression classifier
    classifier = RandomForestClassifier(n_estimators = n_estimators,
                                        criterion = 'entropy',
                                       ) 

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_val)
    
    val_acc = accuracy_score(y_val, y_pred, normalize=True)
    val_f1 = f1_score(y_val, y_pred, average='weighted')
    
    return {'val_acc': val_acc,
            'val_f1': val_f1}

### fitness_rfc ###


def fitness_mnist_dense(data: ga.Data, crit: ga.Criterion):
    # fetch the ML data
    X_train = data.X_train
    X_val = data.X_val
    y_train = data.y_train
    y_val = data.y_val
    
    # get the NN configaration data
    verbose = data.data_dict['verbose']
    epochs = data.data_dict['n_epochs']
    batch_size = data.data_dict['batch_size']
    layer_sizes = data.data_dict['layers']

    model = Sequential()

    input_nodes = data.X_train.shape[1]
    output_nodes = data.y_train.shape[1]
    
    for units in layer_sizes:
        model.add(Dense(units, 
                        kernel_initializer='glorot_normal', 
                        input_dim=input_nodes, 
                        activation='relu')
                    )

        model.add(BatchNormalization())
        
    model.add(Dense(output_nodes, kernel_initializer='glorot_normal', activation='softmax'))

    adam = Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    if verbose >= 1:
        model.summary()

    #try:
    hist = model.fit(X_train, y_train,   
                     validation_data=(X_val, y_val), 
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=verbose,
                    )

    #best_model = load_model('vgg16_1.h5')
    y_pred = model.predict(X_val) # .astype(y_val.dtype)
    
    val_label = np.argmax(y_val, axis=1) # act_label = 1 (index)
    pred_label = np.argmax(y_pred, axis=1) # pred_label = 1 (index)

    val_acc = accuracy_score(val_label, pred_label, normalize=True)
    val_f1 = f1_score(val_label, pred_label, average='weighted')
    
    return {'val_acc': val_acc,
            'val_f1': val_f1}

### fitness_mnist_dense ###


def prepare_data_mnist_cnn(X, y, split_fraction: float):

    return

def fitness_mnist_cnn(data: ga.Data, crit: ga.Criterion):
    # fetch the ML data
    X_train = data.X_train
    X_val = data.X_val
    y_train = data.y_train
    y_val = data.y_val
    
    # get the NN configaration data
    verbose = data.data_dict['verbose']
    epochs = data.data_dict['n_epochs']
    batch_size = data.data_dict['batch_size']
    layers = data.data_dict['layers']
    kernels = data.data_dict['kernels']

    model = Sequential()

    input_nodes = data.X_train.shape[1]
    output_nodes = data.y_train.shape[1]
    
    for i in range(len(layers)):
        layer_size = layers[i]
        kernel_size = kernels[i]

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

        # due to the nature of creating layersizes and kernel sizes quite some settings
        # lead to an impossible configuration. In that case a crash occurs and 
        # generating new layers should be stopped.
        # Caveat: when an exception occurs for some other reason it will be ignored, just
        # output the description of the exception
        except Exception as e:
            logger.debug(f'*** Error -> Layer: {i}: layer_size {layer_size}, kernel_size {kernel_size}')
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

    if verbose >= 1:
        model.summary()

    hist = model.fit(X_train, y_train,   
                     validation_data = (X_val, y_val), 
                     epochs = epochs,
                     batch_size = batch_size,
                     verbose = verbose,
                    )

    #best_model = load_model('vgg16_1.h5')
    y_pred = model.predict(X_val) # .astype(y_val.dtype)
    
    val_label = np.argmax(y_val, axis=1) # act_label = 1 (index)
    pred_label = np.argmax(y_pred, axis=1) # pred_label = 1 (index)

    val_acc = accuracy_score(val_label, pred_label, normalize=True)
    val_f1 = f1_score(val_label, pred_label, average='weighted')

    return {'val_acc': val_acc,
            'val_f1': val_f1}

### fitness_mnist_cnn ###


def analyse_mnist_log_reg(X, y):
    # Rescale the predictors for better performance
    X = X / 255

    split_fraction = 60000

    kick = {
            'max_kicks': 2,
            'generation': 2,
            'trigger': 0.01,
            'keep': 1,
            'p_mutation': 0.25,
            'p_crossover': 10,
           }

    controls = {
                'p_mutation': 0.4,
                'p_crossover': 2,
                'keep': 4,
                'kick': kick,
               }

    variables = {
                 'verbose': 0,
                 'c': 1,
                 'max_iter': 100,
                }

    def variables_ga(pop: ga.Population, data: ga.Data):
        pop.add_var_float('C', 32, 0.01, 10)    

        return


    fitnesses = ['cpu', 'val_acc', 'val_f1', 'f1/cpu', 'doa', 'same']
    criterion = ga.Criterion(fitnesses, fitnesses[2], 'ge', 1.0)

    winners = ga.run(X, y, 
                  population_size = 10,
                  iterations = 50, 
                  prepare_data = prepare_data_mnist_rfc,
                  method = ga.METHOD_ROULETTE,
                  fitness_function = fitness_log_reg,
                  controls = controls,
                  variables = variables, 
                  split_fraction = split_fraction, 
                  pop_variables = variables_ga,
                  criterion = criterion,
                  verbose = 0,
                 )

    return

### analyse_mnist_log_reg ###                


def analyse_mnist_rfc(X, y):
    split_fraction = 60000

    kick = {
            'max_kicks': 2,
            'generation': 2,
            'trigger': 0.01,
            'keep': 1,
            'p_mutation': 0.25,
            'p_crossover': 5,
           }

    controls = {
                'p_mutation': 0.4,
                'p_crossover': 2,
                'keep': 4,
                'kick': kick,
               }

    variables = {
                 'verbose': 1,
                 'n_estimators': 10,
                }

    def variables_ga(pop: ga.Population, data: ga.Data):
        pop.add_var_int('n_estimators', 2, 1000)    

        return


    fitnesses = ['cpu', 'val_acc', 'val_f1', 'f1/cpu', 'doa', 'same']
    criterion = ga.Criterion(fitnesses, fitnesses[2], 'ge', 1.0)

    winners = ga.run(X, y, 
                  population_size = 10,
                  iterations = 50, 
                  prepare_data = prepare_data_mnist_rfc,
                  method = ga.METHOD_ROULETTE,
                  fitness_function = fitness_rfc,
                  controls = controls,
                  variables = variables, 
                  split_fraction = split_fraction, 
                  pop_variables = variables_ga,
                  criterion = criterion,
                  verbose = variables['verbose'],
                 )

    return

### analyse_mnist_rfc ###                


def analyse_mnist_dense(X, y):
    # Rescale the predictors for better performance
    X = X / 255

    # and one hot encode y
    y = OneHotEncoder (sparse=False).fit_transform (y).astype(np.int32)

    split_fraction = 60000

    kick = {
            'max_kicks': 2,
            'generation': 2,
            'trigger': 0.01,
            'keep': 1,
            'p_mutation': 0.25,
            'p_crossover': 10,
           }

    controls = {
                'p_mutation': 0.2,
                'p_crossover': 2,
                'keep': 4,
                'kick': kick,
                }

    variables = {
                 'verbose': 0,
                 'n_epochs': 25,
                 'batch_size': 128,
                 'max_layer_size': 16,
                 'n_layers': 3,
                 'layers': [5, 3, 3, 5]
                }

    def variables_ga(pop: ga.Population, data: ga.Data):
        max_layer_size = data.data_dict['max_layer_size']
        pop.add_var_int('n_layers', 1, 3)
        pop.add_var_int_array('layers', 3, max_layer_size, 'n_layers')

        return

    fitnesses = ['cpu', 'val_acc', 'val_f1', 'f1/cpu', 'doa', 'same']
    criterion = ga.Criterion(fitnesses, fitnesses[2], 'ge', 1.0)

    winners = ga.run(X, y, 
                  population_size = 20,
                  iterations = 50, 
                  prepare_data = prepare_data_mnist_dense,
                  method = ga.METHOD_ROULETTE,
                  fitness_function = fitness_mnist_dense,
                  controls = controls,
                  variables = variables, 
                  split_fraction = split_fraction, 
                  pop_variables = variables_ga,
                  criterion = criterion,
                  verbose = 0,
                )

    return

### analyze_mnist_dense ###                


def analyse_mnist_cnn(X, y):
    split_fraction = 0.25

    kick = {
            'max_kicks': 2,
            'generation': 2,
            'trigger': 0.01,
            'keep': 1,
            'p_mutation': 0.25,
            'p_crossover': 10,
           }

    controls = {
                'p_mutation': 0.2,
                'p_crossover': 2,
                'keep': 4,
                'kick': kick,
                }

    variables = {
                 'verbose': 0,
                 'layer_sizes': [32, 64, 128, 256, 512, 1024],
                 'kernel_sizes': [3, 5, 7],
                 'n_epochs': 25,
                 'batch_size': 128,
                 'n_layers': 3,
                 'layers': [32],
                 'kernels': [3]
                 }

    def variables_ga(pop: ga.Population, data: ga.Data):
        layer_sizes = data.data_dict['layer_sizes']
        kernel_sizes = data.data_dict['kernel_sizes']

        pop.add_var_int('n_layers', 1, 4)
        pop.add_var_int_array('layers', 'I', length='n_layers', value_list=layer_sizes)
        pop.add_var_int_array('kernels', 'I', length='n_layers', value_list=kernel_sizes)

        return


    fitnesses = ['cpu', 'val_acc', 'val_f1', 'f1/cpu', 'doa', 'same']
    criterion = ga.Criterion(fitnesses, fitnesses[2], 'ge', 1.0)

    winners = ga.run(X, y, 
                  population_size = 20,
                  iterations = 50, 
                  prepare_data = prepare_data_mnist_cnn,
                  method = ga.METHOD_ROULETTE,
                  fitness_function = fitness_mnist_cnn,
                  controls = controls,
                  variables = variables, 
                  split_fraction = split_fraction, 
                  pop_variables = variables_ga,
                  criterion = criterion,
                  verbose = 0,
                 )

    return

### analyse_mnist_cnn ###    


# load a simple data set
data = ga.load_data('mnist')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values

analyse_mnist_log_reg(X, y)
#analyse_mnist_rfc(X, y)
#analyse_mnist_dense(X, y)
#analyse_mnist_cnn(X, y)

logger.info('[Ready]')
