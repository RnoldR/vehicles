import os
import time
import pygame
import random
import numpy as np
import pandas as pd

from grid_viewer import GridView2D
from grid import Grid, GridGenerator

from grid_vehicles import Wall, Vehicle, Mushroom, Cactus, Rock, \
    Start, Destination, Dot_green

# Code initialisatie: logging
import logging
import importlib
importlib.reload(logging)

# create logger
logger = logging.getLogger('distances')

logger.setLevel(10)

# create file handler which logs even debug messages
fh = logging.FileHandler('grid-view-2D.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(logging.Formatter('%(message)s'))

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# Initialize Pandas  display options such that the whole DataFrame is printed
pd.options.display.max_rows = 999999
pd.options.display.max_columns = 999999

class Generator(GridGenerator):
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
    
## Class: Generator ##

def test_move_around(res_path: str, icon_style: int):
    things = Grid.load_config(os.path.join(res_path, 'config/config.yaml'))
    things = Grid.load_resources(things, os.path.join(res_path, 'images'), icon_style)
    
    # add a column to things containing the class definitions
    things['Class'] = None
    things.loc['Wall', 'Class'] = Wall
    things.loc['Vehicle', 'Class'] = Vehicle
    things.loc['Mushroom', 'Class'] = Mushroom
    things.loc['Cactus', 'Class'] = Cactus
    things.loc['Rock', 'Class'] = Rock
    things.loc['Start', 'Class'] = Start
    things.loc['Destination', 'Class'] = Destination
    
    print(things)
    
    # create a generator for this test
    generator = Generator(n_mushrooms=5, n_cactuses=4, n_rocks=3)

    # create a grid with appropriate number of columns and rows
    rows = 15
    cols = 20
    grid = Grid(generator, grid_size=(cols, rows), definitions=things)

    # define a grid viewer for the grid
    grid_viewer = GridView2D(grid, things, screen_size=(700, 700))
    
    logger.info(grid.print_grid(grid.grid_cells))
    path = grid.find_route()[1:-1]
    logger.info(path)
    for location in path:
        grid.insert_thing(Dot_green, location)
        
    grid_viewer.update_screen()
    grid_viewer.direction = "X"
    energy = grid.get_vehicles_energy(Vehicle)
    time.sleep(1)
    try:
        while not grid_viewer.game_over and energy > 0:
            grid_viewer.get_events()
            grid_viewer.move_things()
            grid_viewer.update_screen()
            energy = grid.get_vehicles_energy(Vehicle)
   
    finally:
        time.sleep(2)
        pygame.quit()
        
    return
    
if __name__ == "__main__":
    res_path='/media/i-files/home/arnold/development/python/ml/vehicles'

    test_move_around(res_path, 1)
