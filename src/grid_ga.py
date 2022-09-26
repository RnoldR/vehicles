
import sys
import random
import numpy as np
import pandas as pd

from math import sqrt

from grid import Grid, COMPASS
from grid_thing import Sensor, Thing

from grid_objects import Wall, Vehicle, Mushroom, Cactus, Rock, \
    Start, Destination, Dot_green

from grid_thing_data import ICON_STYLE, COL_CATEGORY, COL_ENERGY, COL_ICON, COL_CLASS

# Code initialisatie: logging
import logging
logger = logging.getLogger()

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


