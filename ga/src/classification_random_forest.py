# Random Forest Classification
from create_log import create_logger
logger = create_logger('random-forest-classifier.log')

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from matplotlib.colors import ListedColormap

import ga
from ga import run


def read_adv_data():
    # load a simple data set
    dataset = pd.read_csv('../data/Social_Network_Ads.csv')
    
    # Encode the strings denoting gender to int
    le = LabelEncoder()
    dataset['Gender'] = le.fit_transform(dataset['Gender'])

    # select the data to be trained at
    X = dataset.iloc[:, [1, 2, 3]].values
    y = dataset.iloc[:, 4].values

    return X, y

### read_adv_data ###


def prepare_data_adv_rfc(X, y, split_fraction: float):
    # split in training and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = split_fraction) 

    # Rescale the predictors for better performance
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_val = sc_X.transform(X_val)

    return X_train, X_val, None, y_train, y_val, None

### prepare_data_rfc ###    


def read_mnist_data():
    file_name = '/media/i-files/data/mnist/train.csv'
    logger.info(f'Reading data from: {file_name}')

    raw = pd.read_csv(file_name, sep = ',', header=None, index_col=None).to_numpy()
    X = raw[:, 1:]
    y = raw[:, :1]

    logger.info('Raw.shape = ' + str(raw.shape) + 'type = ' + str(raw.dtype))
    logger.info('X.shape = ' + str(X.shape) + 'type = ' + str(X.dtype))
    logger.info('y.shape = ' + str(y.shape) + 'type = ' + str(y.dtype))
    
    return X, y

### read_mnist_data ###


def prepare_data_mnist_rfc(X, y, split_fraction: float):
    # reshape y
    y = y.reshape(y.shape[0],)

    # split in training and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = split_fraction) 

    # Rescale the predictors for better performance
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_val = sc_X.transform(X_val)

    return X_train, X_val, None, y_train, y_val, None

### prepare_data_mnist_rfc ###    


def fitness_rfc(data: ga.GaData, criterion: ga.Criterion):
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


def analyse_adv_rfc(X, y):
    split_fraction = 0.25

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

    def variables_ga(pop: ga.Population, data: ga.GaData):
        pop.add_var_int('n_estimators', 2, 1000)    

        return


    fitnesses = ['cpu', 'val_acc', 'val_f1', 'f1/cpu', 'doa', 'same']
    criterion = ga.Criterion(fitnesses, fitnesses[2], 'ge', 1.0)

    winners = ga.run(X, y, 
                  population_size = 10,
                  iterations = 50, 
                  prepare_data = prepare_data_adv_rfc,
                  method = ga.METHOD_ROULETTE,
                  fitness_function = fitness_rfc,
                  controls = controls,
                  variables = variables, 
                  split_fraction = split_fraction, 
                  pop_variables = variables_ga,
                  criterion = criterion,
                  verbose = 1,
                 )

    return

### analyse_adv_rfc ###                


def analyse_mnist_rfc(X, y):
    split_fraction = 0.25

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

    def variables_ga(pop: ga.Population, data: ga.GaData):
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


if __name__ == "__main__":
    # Set as many random states as possible
    random_state = 42
    random.seed (random_state)
    np.random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str (random_state)

    # X, y = read_adv_data()
    # analyse_adv_rfc(X, y)
    X, y = read_mnist_data()
    analyse_mnist_rfc(X, y)
