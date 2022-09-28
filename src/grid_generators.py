# import logging
from create_logger import create_logger
logger = create_logger.create_log('grid-vehicles.log')

import os
import time
import pygame
import random
import numpy as np
import pandas as pd

from grid_viewer import GridView2D
from grid import Grid, GridGenerator

from grid_thing_data import ICON_STYLE, COL_CATEGORY, COL_ICON, COL_CLASS

from grid_objects import Wall, Vehicle, Mushroom, Cactus, Rock, Start, Destination, DotGreen

# Initialize Pandas  display options such that the whole DataFrame is printed
pd.options.display.max_rows = 999999
pd.options.display.max_columns = 999999


class RandomGenerator(GridGenerator):
    def __init__(self, n_mushrooms: int, n_cactuses: int, n_rocks: int):
        self.n_mushrooms = n_mushrooms
        self.n_cactuses = n_cactuses
        self.n_rocks = n_rocks
        
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
            
        mid_x = int(grid.grid_size[0] / 2)
        mid_y = int(grid.grid_size[1] / 2)
        
        for y in range(2, grid.grid_size[1] - 2):
            grid.insert_thing(Wall, (mid_x, y))
            
        for x in range(1, mid_x -2):
            grid.insert_thing(Wall, (x, mid_y))
            
        for x in range(mid_x, grid.grid_size[0] -2):
            grid.insert_thing(Wall, (x, 3))
            
        # Start upper left and destination lower right
        self.init_pos: tuple = (1, 1)
        grid.START = grid.insert_thing(Start, self.init_pos)
        grid.DESTINATION = grid.insert_thing(Destination, (grid.grid_size[0]-2, grid.grid_size[1]-2))
        
        # set vehicle on the start position and have it tracked by the grid
        vehicle_to_be_tracked = grid.insert_thing(Vehicle, self.init_pos)
        grid.set_tracker(vehicle_to_be_tracked)
        
        grid.insert_things(Mushroom, grid.generate_random_locs(self.n_mushrooms))
        grid.insert_things(Cactus, grid.generate_random_locs(self.n_cactuses))
        grid.insert_things(Rock, grid.generate_random_locs(self.n_rocks))

        return
    
    # generate #
    
### Class: RandomGenerator ###


class FixedGenerator(GridGenerator):
    def __init__(self, level: int = 1):
        self.level: int = level

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

        if self.level == 1:
            self.level_1(grid)

        elif self.level == 2:
            self.level_2(grid)

        elif self.level == 3:
            self.level_3(grid)

        elif self.level == 4:
            self.level_4(grid)

        else:
            raise ValueError('Wrong level specified, should be in range [1..3]')

        # if

        # create start upper left and destination lower right
        self.init_pos: tuple = (1, 1)
        grid.set_start(Start, self.init_pos)
        grid.set_destination(Destination, (grid.grid_size[0]-2, grid.grid_size[1]-2))

        return
    
    # generate #

    def level_1(self, grid: Grid):
        # create cactuses
        self.create_things(grid, Cactus, 
                           x_start = 1, y_start = 1,
                           x_incr = 6, y_incr = 3, n_incr = 3)

        # create mushrooms
        self.create_things(grid, Mushroom, 
                           x_start = 4, y_start = 3,
                           x_incr = 6, y_incr = 3, n_incr = 3)

        return

    ### level_1 ###

    def level_2(self, grid: Grid):
        # create cactuses
        self.create_things(grid, Cactus, 
                           x_start = 1, y_start = 1,
                           x_incr = 6, y_incr = 2, n_incr = 3)

        # create mushrooms
        self.create_things(grid, Mushroom, 
                           x_start = 4, y_start = 3,
                           x_incr = 6, y_incr = 3, n_incr = 3)

        return

    ### level_2 ###

    def level_3(self, grid: Grid):
        # create cactuses
        self.create_things(grid, Cactus, 
                           x_start = 1, y_start = 1,
                           x_incr = 2, y_incr = 2, n_incr = 3)

        # create mushrooms
        self.create_things(grid, Mushroom, 
                           x_start = 4, y_start = 3,
                           x_incr = 6, y_incr = 3, n_incr = 3)

        return

    ### level_3 ###

    def level_4(self, grid: Grid):
        # create cactuses
        self.create_things(grid, Cactus, 
                           x_start = 1, y_start = 1,
                           x_incr = 2, y_incr = 2, n_incr = 3)

        return

    ### level_4 ###

    def create_things(self, grid: Grid, ThingClass, 
                      x_start: int, y_start: int, 
                      x_incr: int, y_incr: int, n_incr: int):
        y = y_start
        n = 0
        while y < grid.grid_size[1] - 1:
            n %= n_incr
            n += 1
            x = x_start

            while x < grid.grid_size[0] - 1:
                grid.insert_thing(ThingClass, (x, y))

                x += x_incr

            # while

            y += y_incr

        # while
            
        return

    ### create_things ###        
    
### Class: FixedGenerator ###