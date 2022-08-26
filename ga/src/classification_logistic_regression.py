from create_log import create_logger
logger = create_logger('logistic-regression-classifier.log')

import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from matplotlib.colors import ListedColormap

from tensorflow.keras.datasets import mnist

from ga import Population, Criterion
from ga import GaData as Data

# Code initialisatie: logging

def plot(classifier, X_set, y_set):
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
    # for

    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.draw()
    
    return


def fitness_glm(data: Data, crit: Criterion):
    cpu = time.time()
    
    # fetch the ML data
    X_train = data.X_train
    X_val = data.X_val
    y_train = data.y_train
    y_val = data.y_val

    # fetch the parameters for the logistic regression from data
    C = data.data_dict['C']
    max_iter = data.data_dict['max_iter']
    
    # Create a logistic regression classifier
    classifier = LogisticRegression(C=C, max_iter=max_iter)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_val)

    cpu = time.time() - cpu
    
    val_acc = accuracy_score(y_val, y_pred, normalize=True)
    val_f1 = f1_score(y_val, y_pred, average='weighted')

    return {'val_acc': val_acc,
            'val_f1': val_f1}


def pretest(Cs):
    ##### test #####
    for c in Cs:
        classifier = LogisticRegression(C=c)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred, normalize=True)
        val_f1 = f1_score(y_val, y_pred, average='weighted')
        logger.info(f'Test for C = {c}: acc {val_acc}, F1 {val_f1}')

    return


def classify_adv(fitnesses, criterion):
    # load a simple data set
    dataset = pd.read_csv('../data/Social_Network_Ads.csv')

    # Encode the strings denoting gender to int
    le = LabelEncoder()
    dataset['Gender'] = le.fit_transform(dataset['Gender'])

    # select the data to be trained at
    X = dataset.iloc[:, [1, 2, 3]].values
    y = dataset.iloc[:, 4].values

    # split in training and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, 
                                                      random_state = seed)

    # Rescale the predictors for better performance
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_val = sc_X.transform(X_val)

    logger.info('X_train.shape = ' + str(X_train.shape) + 'type = ' + str(X_train.dtype))
    logger.info('y_train.shape = ' + str(y_train.shape) + 'type = ' + str(y_train.dtype))
    logger.info('X_val.shape =   ' + str(X_val.shape) + 'type = ' + str(X_val.dtype))
    logger.info('y_val.shape =   ' + str(y_val.shape) + 'type = ' + str(y_val.dtype))

    #pretest([0.0001, 0.1, 0.5, 0.99])
    #sys.exit()
    # set parameter for log regression
    c = 0.01
    max_iter = 100
    data = Data(X_train, X_val, None, y_train, y_val, None)
    data.register_variable('C', c)
    data.register_variable('max_iter', max_iter)

    fitness = fitness_glm(data, criterion)
    logger.info('validation accuracy: {:.2f}'.format(fitness['val_acc']))
    logger.info('validation F1:       {:.2f}'.format(fitness['val_f1']))

    pop = Population(p_mutation=0.2, 
                     p_crossover=2, # > 1 means absolute # of crossovers 
                     fitness=fitnesses, 
                     selection_key=criterion.selection_key)

    pop.add_var_float('C', 32, 0.0001, 0.5)    
    pop.set_fitness_function(fitness_glm, data)
    pop.create_population(10)

    logger.warning('')
    logger.warning('--- Generation 0 ---')
    pop.pre_compute(criterion)
    pop.show()

    for generation in range(1, 2):
        logger.warning('')
        logger.warning('*** Generation ' + str(generation))
        pop.next_generation(10, criterion)
        pop.show()
    # for

    return    


def classify_mnist(fitnesses, criterion):
    from tensorflow.keras.datasets import mnist

    # load mnist data set
    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    
    # reshape X to fit for CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    #y_train = y_train.reshape(y_train.shape[0], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
    #y_val = y_val.reshape(y_val.shape[0], 1)

    # normalize X
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    # and one hot encode y
    #y_train = OneHotEncoder(sparse=False).fit_transform (y_train).astype(np.int32)
    #y_val = OneHotEncoder(sparse=False).fit_transform (y_val).astype(np.int32)

    logger.info('X_train.shape = ' + str(X_train.shape) + 'type = ' + str(X_train.dtype))
    logger.info('y_train.shape = ' + str(y_train.shape) + 'type = ' + str(y_train.dtype))
    logger.info('X_val.shape =   ' + str(X_val.shape) + 'type = ' + str(X_val.dtype))
    logger.info('y_val.shape =   ' + str(y_val.shape) + 'type = ' + str(y_val.dtype))

    # set parameter for log regression
    c = 0.01
    max_iter = 1000
    data = Data(X_train, X_val, None, y_train, y_val, None)
    data.register_variable('C', c)
    data.register_variable('max_iter', max_iter)

    fitness = fitness_glm(data, criterion)
    logger.info('validation accuracy: {:.2f}'.format(fitness['val_acc']))
    logger.info('validation F1:       {:.2f}'.format(fitness['val_f1']))

    pop = Population(p_mutation=0.2, 
                     p_crossover=2, # > 1 means absolute # of crossovers 
                     fitness=fitnesses, 
                     selection_key=criterion.selection_key)

    pop.add_var_float('C', 32, 0.0001, 0.5)    
    pop.set_fitness_function(fitness_glm, data)
    pop.create_population(10)

    logger.warning('')
    logger.warning('--- Generation 0 ---')
    pop.pre_compute(criterion)
    pop.show()

    for generation in range(1, 2):
        logger.warning('')
        logger.warning('*** Generation ' + str(generation))
        pop.next_generation(10, criterion)
        pop.show()
      
    return


fitnesses = ['cpu', 'val_acc', 'acc/cpu'] # 'val_acc', 'val_f1', 'f1/cpu']
criterion = Criterion(fitnesses[2], 'le', 1.0)

if __name__ == "__main__":
    logger.info(' ')
    logger.info('=========================================')
    
    seed = 42
    random.seed(seed)

    classify_adv(fitnesses, criterion)