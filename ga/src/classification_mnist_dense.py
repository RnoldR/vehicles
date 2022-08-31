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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

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


def read_data():
    file_name = '/media/i-files/data/mnist/train.csv'
    logger.info(f'Reading data from: {file_name}')
    raw = pd.read_csv(file_name, sep = ',', header=None, index_col=None).to_numpy()
    X = raw[:, 1:]
    y = raw[:, :1]
    logger.info('Raw.shape = ' + str(raw.shape) + 'type = ' + str(raw.dtype))
    logger.info('X.shape = ' + str(X.shape) + 'type = ' + str(X.dtype))
    logger.info('y.shape = ' + str(y.shape) + 'type = ' + str(y.dtype))
    
    return X, y

### read_data ###


def classify_mnist_dense(data: ga.GaData, crit: ga.Criterion):
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

### classify_mnist_dense ###


def classify_mnist_cnn(data: ga.GaData, crit: ga.Criterion):
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

### classify_mnist_cnn ###


def run(X, y, 
        population_size: int = 10,
        iterations: int = 10,
        split_fraction: float = 0.33,
        method: str = 'elite',
        fitness_function = None,
        variables: dict = None, 
        pop_variables = None, 
        criterion: ga.Criterion = None,
       ):

    # and one hot encode y
    y = OneHotEncoder (sparse=False).fit_transform (y).astype(np.int32)

    # split in training and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = split_fraction, 
                                                      random_state = random_state)

    data = ga.GaData(X_train, X_val, None, y_train, y_val, None)
    data.register(variables)

    kick = {'max_kicks': 2,
            'generation': 10,
            'trigger': 0.01,
            'keep': 1,
            'p_mutation': 0.25,
            'p_crossover': 10,
           }

    pop = ga.Population(p_mutation=0.02, 
                        p_crossover=2, # > 1 means absolute # of crossovers 
                        method = method,
                        fitness = criterion.fitnesses,
                        selection_key = criterion.selection_key,
                        kick = kick,
                        keep = 5,
                        best_of = 0,
                        random_state = 42,
                    )

    pop_variables(pop, data)

    logger.info('')
    logger.info(f'Creating a population of size {population_size}. This may take quite some time.')
    pop.set_fitness_function(fitness_function, data)
    pop.create_population(population_size, criterion)

    logger.info('')
    logger.info('--- Initial Generation ---')
    pop.show(show_bits = False)

    i = 0
    while pop.next_generation(population_size, criterion):
        pop.show(show_bits = True)

        if i == iterations:
            break

        else:
            i += 1

        # if
    # while

    gens, tops = pop.statistics(criterion.selection_key, 'top')
    gens, means = pop.statistics(criterion.selection_key, 'mean')
    gens, sds = pop.statistics(criterion.selection_key, 's.d.')

    plt.plot(gens, tops, color='g', label='top')
    plt.plot(gens, means, color='r', label='mean')
    plt.plot(gens, sds, color='b', label='sd')
    plt.title(criterion.selection_key)
    plt.legend()
    plt.show()

    return pop.population

### run ###


def analyse_mnist_dense(X, y):
    # Rescale the predictors for better performance
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_val = sc_X.transform(X_val)

    split_fraction = 0.25

    variables = {
                'verbose': 0,
                'n_epochs': 25,
                'batch_size': 128,
                'max_layer_size': 16,
                'n_layers': 3,
                'layers': [5, 3, 3, 5]
                }

    def variables_dense(pop: ga.Population, data: ga.GaData):
        max_layer_size = data.data_dict['max_layer_size']
        pop.add_var_int('n_layers', 1, 3)
        pop.add_var_int_array('layers', 3, max_layer_size, 'n_layers')

    fitnesses = ['cpu', 'val_acc', 'val_f1', 'f1/cpu', 'doa', 'same']
    criterion = ga.Criterion(fitnesses, fitnesses[2], 'ge', 1.0)

    winners = run(X, y, 
                population_size = 10,
                iterations = 10, 
                method = ga.METHOD_ROULETTE,
                fitness_function = classify_mnist_dense,
                variables = variables, 
                split_fraction = split_fraction, 
                pop_variables = variables_dense,
                criterion = criterion,
                )

    return

### analyze_mnist_dense ###                


def analyse_mnist_cnn(X, y):
    # reshape X to fit for CNN
    X = X.reshape(X.shape[0], 28, 28, 1)
    y = y.reshape(y.shape[0], 1)

    # normalize X
    X = X / 255.0

    split_fraction = 0.25

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

    def variables_cnn(pop: ga.Population, data: ga.GaData):
        layer_sizes = data.data_dict['layer_sizes']
        kernel_sizes = data.data_dict['kernel_sizes']

        pop.add_var_int('n_layers', 1, 4)
        pop.add_var_int_array('layers', 'I', length='n_layers', value_list=layer_sizes)
        pop.add_var_int_array('kernels', 'I', length='n_layers', value_list=kernel_sizes)


    fitnesses = ['cpu', 'val_acc', 'val_f1', 'f1/cpu', 'doa', 'same']
    criterion = ga.Criterion(fitnesses, fitnesses[2], 'ge', 1.0)

    winners = run(X, y, 
                  population_size = 10,
                  iterations = 10, 
                  method = ga.METHOD_ROULETTE,
                  fitness_function = classify_mnist_cnn,
                  variables = variables, 
                  split_fraction = split_fraction, 
                  pop_variables = variables_cnn,
                  criterion = criterion,
                 )

    return

### analyse_mnist_cnn ###    


# load a simple data set
X, y = read_data()

#analyse_mnist_dense(X, y)
analyse_mnist_cnn(X, y)

logger.info('[Ready]')
