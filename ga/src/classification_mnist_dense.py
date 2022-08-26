#import logging
#from logging.config import dictConfig



# # Code initialisatie: logging
# # create logger
# LOGGING = { 
#     'version': 1,
#     'disable_existing_loggers': False,
#     'formatters': { 
#         'standard': { 
#             'format': '%(asctime)s [%(levelname)s] %(module)s: %(message)s'
#         },
#         'brief': {
#             'format': '%(message)s'
#         },
#     },
#     'handlers': { 
#         'console': { 
#             'level': 'INFO',
#             'formatter': 'brief',
#             'class': 'logging.StreamHandler',
#             'stream': 'ext://sys.stdout',  # Default is stderr
#         },
#         'file': { 
#             'level': 'DEBUG',
#             'formatter': 'standard',
#             'class': 'logging.FileHandler',
#             'filename': 'mnist_classifier.log', 
#             'mode': 'w',
#         },
#     },
#     'loggers': {
#         '': {
#             'level': 'INFO',
#             'handlers': ['console', 'file']
#         },
#     },    
# }

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logging.config.dictConfig(LOGGING)

# logger.info('*** MNIST classification')

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, f1_score

import ga

#printout()

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


def classify_mnist(data: ga.GaData, crit: ga.Criterion):
    cpu = time.time()
    
    # fetch the ML data
    X_train = data.X_train
    X_val = data.X_val
    y_train = data.y_train
    y_val = data.y_val
    
    # get the NN configaration data
    verbose = data.data_dict['verbose']
    epochs = data.data_dict['n_epochs']
    batch_size = data.data_dict['batch_size']
    n_layers = data.data_dict['n_layers']
    layer_sizes = data.data_dict['layers']
    #for i in range(n_layers):
    #    layer_sizes.append(data.data_dict[f'layer_size_{i:d}'])

    model = Sequential()

    input_nodes = data.X_train.shape[1]
    output_nodes = data.y_train.shape[1]
    
    lagen = []
    # print('Layers:', layer_sizes)
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
    f1 = f1_score(val_label, pred_label, average='weighted')
    acc_cpu = val_acc / cpu

    #except:
    #val_acc = -1
    #f1 = -1
    #acc_cpu = -1        
    
    # try..except
    
    return [val_acc, f1, acc_cpu]

### classify_mnist ###


# load a simple data set
X, y = read_data()

# normalize X
X = X / 255.0

# and one hot encode y
y = OneHotEncoder (sparse=False).fit_transform (y).astype(np.int32)

# split in training and test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, 
                                                  random_state = random_state)

# Rescale the predictors for better performance
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_val = sc_X.transform(X_val)

# set parameter for neural network and ga
pop_size: int = 5
n_epochs: int = 10
n_layers: int = 4
layer_size: list = [3, 5, 5, 3, 5, 6]
max_layer_size = 63
batch_size = 128

fitnesses = ['val_acc', 'val_f1', 'val/cpu']
pop = ga.Population(p_mutation=0.25, p_crossover=5, 
                    fitness=fitnesses, selection_key=fitnesses[1])

data = ga.GaData(X_train, X_val, None, y_train, y_val, None)
data.register_variable('verbose', 1)
data.register_variable('n_epochs', n_epochs)
data.register_variable('batch_size', batch_size)
data.register_variable('n_layers', n_layers)
data.register_variable('layers', [5, 3, 3, 5])

crit = ga.Criterion('val_f1', 'ge', 1.0)

pop.add_var_int('n_epochs', 10, 100)
pop.add_var_int('n_layers', 1, 4)
pop.add_var_int_array('layers', 3, max_layer_size, 'n_layers')

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
