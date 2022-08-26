# Random Forest Classification

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

from ga import GaData as Data
from ga import Population

# Code initialisatie: logging
import logging
import importlib
importlib.reload(logging)

# create logger
logger = logging.getLogger('ga')

logger.setLevel(10)

# create file handler which logs even debug messages
fh = logging.FileHandler('ga.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(logging.Formatter('%(message)s'))

def plot(X_set, y_set, classifier):
    cmap = np.array(['red', 'green'])
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1,   step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = cmap[i], label = j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.draw()
    
    return

### plot ###

def classify_rfc(data: Data):
    cpu = time.time()
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
                                        random_state = 0, n_jobs=1)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_val)
    cpu = time.time() - cpu
    
    val_acc = accuracy_score(y_val, y_pred, normalize=True)
    f1 = f1_score(y_val, y_pred, average='weighted')
    acc_cpu = val_acc / cpu
    
    return [val_acc, f1, acc_cpu]

### classify_rfc ###

if __name__ == "__main__":
    logger.warning(' ')
    logger.warning('=========================================')

    seed = 42
    random.seed(seed)

    # load a simple data set
    dataset = pd.read_csv('../data/Social_Network_Ads.csv')
    
    # Encode the strings denoting gender to int
    le = LabelEncoder()
    dataset['Gender'] = le.fit_transform(dataset['Gender'])

    # select the data to be trained at
    X = dataset.iloc[:, [1, 2, 3]].values
    y = dataset.iloc[:, 4].values

    # split in training and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, 
                                                      random_state = seed)

    # Rescale the predictors for better performance
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_val = sc_X.transform(X_val)

    # set parameter for log regression
    n_estimators = 10
    data = Data(X_train, X_val, None, y_train, y_val, None)
    data.register_variable('n_estimators', n_estimators)
    fitness = classify_rfc(data)
    logger.warning('validation accuracy: {:.2f}'.format(fitness[0]))
    logger.warning('validation F1:       {:.2f}'.format(fitness[1]))
    logger.warning('val accuracy / cpu:  {:.2f}'.format(fitness[2]))

    # Create the population and its parameters
    fitnesses = ['val_acc', 'val_f1', 'acc/cpu']
    pop = Population(p_mutation=0.2, p_crossover=2, 
                     fitness=fitnesses, 
                     selection_key=fitnesses[0])
    
    # add the variables and fitness function
    pop.add_var('n_estimators', 16, 'I', 2, 1000)    
    pop.set_fitness_function(classify_rfc, data)
    pop.create_population(10)
    
    # show the start population
    logger.warning('--- Generation 0 ---')
    pop.pre_compute()
    pop.show()
    
    # loop over n generations
    cpu = time.time()
    for generation in range(1, 10):
        logger.warning('')
        pop.next_generation(10)
        logger.warning('*** Generation {:d} in {:.2f} seconds ***'.
                       format(generation, time.time() - cpu))
        pop.show()
      