from create_log import create_logger
logger = create_logger('logistic-regression-classifier.log')

import sys
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from matplotlib.colors import ListedColormap

#from tensorflow.keras.datasets import mnist

from ga import ga


def fitness_log_reg(data: ga.Data, crit: ga.Criterion):
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

        logger.info(f'Ok val_acc {val_acc:.2f}, val_f1 {val_f1:.2f}')

    # try..except

    return {'val_acc': val_acc,
            'val_f1': val_f1}

### fitness_log_reg ###


def prepare_data_adv_logreg(X, y, split_fraction):
    # Encode the strings denoting gender to int
    le = LabelEncoder()
    X['Gender'] = le.fit_transform(X['Gender'])

    # convert to numpy arrays
    X = X.values
    y = y.values

    # split in training and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size = split_fraction, 
                                                      random_state = seed)

    # Rescale the predictors for better performance
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_val = sc_X.transform(X_val)

    logger.info('X_train.shape = ' + str(X_train.shape) + 'type = ' + str(X_train.dtype))
    logger.info('y_train.shape = ' + str(y_train.shape) + 'type = ' + str(y_train.dtype))
    logger.info('X_val.shape =   ' + str(X_val.shape) + 'type = ' + str(X_val.dtype))
    logger.info('y_val.shape =   ' + str(y_val.shape) + 'type = ' + str(y_val.dtype))

    return X_train, X_val, None, y_train, y_val, None

### prepare_data_adv_logreg ###    


def analyse_adv_logreg():
    data = ga.load_data('Social_Network_Ads')

    # select the data to be trained at
    X = data.loc[:, ['Gender', 'Age', 'EstimatedSalary']]
    y = data.loc[:, 'Purchased']

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
                'p_mutation': 0.02,
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
                  prepare_data = prepare_data_adv_logreg,
                  method = ga.METHOD_ROULETTE,
                  fitness_function = fitness_log_reg,
                  controls = controls,
                  variables = variables, 
                  split_fraction = split_fraction, 
                  pop_variables = variables_ga,
                  criterion = criterion,
                  verbose = 0,
                 )

    gens, tops = winners.statistics(criterion.selection_key, 'top')
    gens, means = winners.statistics(criterion.selection_key, 'mean')
    gens, sds = winners.statistics(criterion.selection_key, 's.d.')

    plt.plot(gens, tops, color='g', label='top')
    plt.plot(gens, means, color='r', label='mean')
    plt.plot(gens, sds, color='b', label='sd')
    plt.title(criterion.selection_key)
    plt.legend()

    plt.show()

    winners.show(show_bits=False)

    return

### analyse_adv_logreg ###


def prepare_data_mnist_logreg(X: pd.DataFrame, y: pd.DataFrame, split_fraction):
    # convert X and y to dataframes
    X = X.values
    y = y.values

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

### prepare_data_mnist_logreg ###    


def analyse_mnist_logreg():
    data = ga.load_data('mnist')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

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
                  iterations = 1000, 
                  prepare_data = prepare_data_mnist_logreg,
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


iters = 20
fitnesses = ['cpu', 'val_f1', 'f1/cpu', 'doa'] 
criterion = ga.Criterion(fitnesses, fitnesses[1], 'le', 1.0)

if __name__ == "__main__":
    logger.info(' ')
    logger.info('=========================================')
    
    seed = 42
    random.seed(seed)

    # load a simple data set
    analyse_adv_logreg()
    analyse_mnist_logreg()
