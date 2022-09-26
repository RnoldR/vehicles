
import sys
import time
import random
import numpy as np
import pandas as pd

from math import sqrt

from grid import Grid, COMPASS
from grid_thing import Sensor, Thing

from ga import ga

from grid_vehicles import Simple

#from grid_thing_data import ICON_STYLE, COL_CATEGORY, COL_ENERGY, COL_ICON, COL_CLASS

# Code initialisatie: logging
import logging
logger = logging.getLogger()


def prepare_data(X, y, split_fraction: float):

    return

### prepare_data ###


def fitness_simple_vehicle(data: ga.Data, criterion: ga.Criterion, generator):
    # fetch the ML data
    X_train = data.X_train
    X_val = data.X_val
    y_train = data.y_train
    y_val = data.y_val
    
    # fetch the parameters for the logistic regression from data
    n_estimators = data.data_dict['n_estimators']
    
    rows = 15
    cols = 20

    definitions = load_thing_definitions(res_path, icon_style)
    
    # create a generator for this test
    #generator = FixedGenerator()

    # create a grid with appropriate number of columns and rows
    grid = Grid(generator(), grid_size=(cols, rows), definitions=definitions)

    logger.info(grid.print_grid(grid.grid_cells))
        
    energy = grid.get_vehicles_energy(Simple)
    time.sleep(1)

    w_wall = data.data_dict['w_wall']
    w_mush = data.data_dict['w_mushroom']
    w_cact = data.data_dict['w_cactus']
    w_dest = data.data_dict['w_target']

    # set vehicle on the start position and have it tracked by the grid
    vehicle_to_be_tracked = grid.insert_thing(Simple, grid.start.location)
    vehicle_to_be_tracked.set_weights(w_wall, w_mush, w_cact, w_dest)
    grid.set_tracker(vehicle_to_be_tracked)

    energy = 1
    try:
        while not grid.destination_reached(1000) and energy > 0:
            grid.next_turn()
            energy = grid.tracked.energy

        # while

    except Exception as e:
        logger.info('Exception: ' + str(e))

    # try..except

    reached = 0
    if grid.destination_reached():
        reached = 1
        logger.info(f'Destination reached in {grid.turns} turns.')

    total = grid.turns
    if reached == 0:
        total *= total
    
    return {'reached': reached,
            'turns': grid.turns,
            'total': total,
           }

### fitness_simple_vehicle ###


def analyse_simple_vehicle(X, y):
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
                 'w_wall': -0.1,
                 'w_mushroom': 0.25,
                 'w_cactus': -0.75,
                 'w_target': 1.0,
                 'verbose': 0,
                }

    def variables_ga(pop: ga.Population, data: ga.Data):
        precision = 16
        pop.add_var_float('w_wall', precision, -1.0, 1.0)    
        pop.add_var_float('w_mushroom', precision, -1.0, 1.0)    
        pop.add_var_float('w_cactus', precision, -1.0, 1.0)    
        pop.add_var_float('w_target', precision, -1.0, 1.0)    

        return

    ### variables_ga ###

    fitnesses = ['cpu', 'reached', 'turns', 'total']
    criterion = ga.Criterion(fitnesses, fitnesses[3], 'le', 1.0)

    winners = ga.run(X, y, 
                  population_size = 10,
                  iterations = 50, 
                  prepare_data = prepare_data,
                  method = ga.METHOD_ROULETTE,
                  fitness_function = fitness_simple_vehicle,
                  controls = controls,
                  variables = variables, 
                  split_fraction = split_fraction, 
                  pop_variables = variables_ga,
                  criterion = criterion,
                  verbose = variables['verbose'],
                 )

    return

### analyse_simple_vehicle ###                
