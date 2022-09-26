
import sys
import time
import random
import numpy as np
import pandas as pd

from math import sqrt

from grid import Grid, GridGenerator

from ga import ga

from grid_objects import Wall, Mushroom, Cactus, Start, Destination
from grid_vehicles import Simple

#from grid_thing_data import ICON_STYLE, COL_CATEGORY, COL_MASS, COL_ICON, COL_CLASS

# Code initialisatie: logging
import logging
logger = logging.getLogger()


class FixedGenerator(GridGenerator):
    def __init__(self):
        
        return
    
    # __init__ #
    
    def generate(self, grid: Grid):
        # Create walls around the grid
        for x in range(grid.grid_size[0]):
            grid.insert_thing(Wall, (x, 0))
            grid.insert_thing(Wall, (x, grid.grid_size[1]-1))
            
        for y in range(grid.grid_size[1]):
            grid.insert_thing(Wall, (0, y))
            grid.insert_thing(Wall, (grid.grid_size[0]-1, y))

        # create catuses
        x_incr = 6
        y_incr = 2
        y = 2
        n = 0
        while y < grid.grid_size[1] - 1:
            n %= 3
            n += 1
            x = n

            while x < grid.grid_size[0] - 1:
                grid.insert_thing(Cactus, (x, y))

                x += x_incr

            # while

            y += y_incr

        # while
            
        # create mushrooms
        y = 3
        n = 0
        while y < grid.grid_size[1] - 1:
            n %= 3
            n += 4
            x = n

            while x < grid.grid_size[0] - 1:
                grid.insert_thing(Mushroom, (x, y))

                x += x_incr

            # while

            y += y_incr

        # while

        # create start upper left and destination lower right
        self.init_pos: tuple = (1, 1)
        grid.set_start(Start, self.init_pos)
        grid.set_destination(Destination, (grid.grid_size[0]-2, grid.grid_size[1]-2))

        return
    
    # generate #
    
### Class: FixedGenerator ###


def prepare_data(X, y, split_fraction: float):

    return None, None, None, None, None, None

### prepare_data ###


def fitness_simple_vehicle(data: ga.Data, criterion: ga.Criterion):
    # fetch the parameters from data
    res_path = data.data_dict['res_path']
    icon_style = data.data_dict['icon_style']
    verbose = data.data_dict['verbose']
    
    rows = 15
    cols = 20

    # create a generator for this test
    #generator = FixedGenerator()

    # create a grid with appropriate number of columns and rows
    grid = Grid(FixedGenerator(), 
                grid_size = (cols, rows), 
                res_path = res_path, 
                icon_style = icon_style,
                verbose = verbose,
               )

    time.sleep(1)

    w_wall = data.data_dict['w_wall']
    w_mush = data.data_dict['w_mushroom']
    w_cact = data.data_dict['w_cactus']
    w_dest = data.data_dict['w_target']

    # set vehicle on the start position and have it tracked by the grid
    vehicle_to_be_tracked = grid.insert_thing(Simple, grid.start.location)
    vehicle_to_be_tracked.set_weights(w_wall, w_mush, w_cact, w_dest)
    grid.set_tracker(vehicle_to_be_tracked)

    mass = grid.get_vehicles_mass(Simple)
    try:
        while not grid.destination_reached(1000) and mass > 0:
            grid.next_turn()
            mass = grid.tracked.mass

        # while

    except Exception as e:
        logger.info('Exception: ' + str(e))

    # try..except

    reached = 0
    if grid.destination_reached():
        reached = 1
        logger.info(f'Destination reached in {grid.turns} turns.')

    mass = grid.get_vehicles_mass(Simple)
    
    return {'reached': reached,
            'turns': grid.turns,
            'mass': mass,
           }

### fitness_simple_vehicle ###


def analyse_simple_vehicle(X, y, res_path: str, icon_style: int):
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
                 'res_path': res_path,
                 'icon_style': icon_style,
                }

    def variables_ga(pop: ga.Population, data: ga.Data):
        precision = 16
        pop.add_var_float('w_wall', precision, -1.0, 1.0)    
        pop.add_var_float('w_mushroom', precision, -1.0, 1.0)    
        pop.add_var_float('w_cactus', precision, -1.0, 1.0)    
        pop.add_var_float('w_target', precision, -1.0, 1.0)    

        return

    ### variables_ga ###

    fitnesses = ['cpu', 'reached', 'turns', 'mass']
    criterion = ga.Criterion(fitnesses, fitnesses[3], 'ge', 1.0)

    winners = ga.run(X, y, 
                  population_size = 20,
                  iterations = 10, 
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

    return winners

### analyse_simple_vehicle ###                
