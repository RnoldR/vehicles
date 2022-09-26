#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from grid_thing import Thing
from grid_objects import Wall, Vehicle, Mushroom, Cactus, Rock, \
    Start, Destination, DotGreen
from grid_vehicles import Simple
from grid_ga import analyse_simple_vehicle

# Initialize Pandas  display options such that the whole DataFrame is printed
pd.options.display.max_rows = 999999
pd.options.display.max_columns = 999999

import random
random.seed(41)

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


def test_move_around(res_path: str, icon_style: int):
    screen_width = 700
    screen_height = 700
    rows = 15
    cols = 20

    # create a generator for this test
    generator = RandomGenerator(n_mushrooms=5, n_cactuses=4, n_rocks=3)

    # create a grid with appropriate number of columns and rows
    grid = Grid(generator, 
                grid_size = (cols, rows), 
                res_path = res_path, 
                icon_style = icon_style
               )


    # define a grid viewer for the grid
    grid_viewer = GridView2D(grid, grid.definitions, screen_size=(screen_width, screen_height))
    
    logger.info(grid.print_grid(grid.grid_cells))
        
    grid_viewer.update_screen()
    grid_viewer.direction = "X"
    mass = grid.get_vehicles_mass(Vehicle)
    time.sleep(1)

    try:
        while not grid_viewer.game_over and mass > 0:
            grid_viewer.get_events()
            grid_viewer.move_things()
            grid_viewer.update_screen()
            mass = grid.get_vehicles_mass(Vehicle)
   
    finally:
        time.sleep(2)
        pygame.quit()
        
    return
    
### test_move_around ###


def test_move_auto(res_path: str, icon_style: int, generator,
                   w_wall = -0.75, w_mush = 0.5, w_cact = -1.0, w_dest = 1.0):
    screen_width = 700
    screen_height = 700
    rows = 15
    cols = 20

    # create a grid with appropriate number of columns and rows
    grid = Grid(generator(), 
                grid_size = (cols, rows), 
                res_path = res_path, 
                icon_style = icon_style,
               )

    logger.info(grid.print_grid(grid.grid_cells))
        
    # set vehicle on the start position and have it tracked by the grid
    vehicle_to_be_tracked = grid.insert_thing(Simple, grid.start.location)
    vehicle_to_be_tracked.set_weights(w_wall, w_mush, w_cact, w_dest)
    vehicle_to_be_tracked.leave_trace = True
    grid.set_tracker(vehicle_to_be_tracked)

    # define a grid viewer for the grid
    grid_viewer = GridView2D(grid, grid.definitions, screen_size=(screen_width, screen_height))
    grid_viewer.update_screen()
    grid_viewer.direction = "X"

    mass = grid.get_vehicles_mass(Vehicle)
    time.sleep(1)

    try:
        while not grid.destination_reached() and not grid_viewer.game_over and mass > 0:
            # grid_viewer.get_events()
            # grid_viewer.move_things()
            grid_viewer.next_turn()
            grid_viewer.update_screen()
            mass = grid.get_vehicles_mass(Vehicle)
            time.sleep(0.25)

    finally:
        if grid.destination_reached():
            logger.info(f'Destination reached in {grid.turns} turns.')
        
        time.sleep(20)
        pygame.quit()

    return
    
### test_move_auto ###


def test_many_vehicles(res_path: str, icon_style: int, generator, n: int) -> int:
    def loop_one_grid(w_wall: float, w_mush: float, w_cact: float, w_dest: float):
        rows = 15
        cols = 20

        # create a grid with appropriate number of columns and rows
        grid = Grid(generator(), 
                    grid_size = (cols, rows), 
                    res_path = res_path, 
                    icon_style = icon_style,
                    verbose = 0
                   )

        logger.info(grid.print_grid(grid.grid_cells))
            
        mass = grid.get_vehicles_mass(Vehicle)
        time.sleep(1)

        # set vehicle on the start position and have it tracked by the grid
        vehicle_to_be_tracked = grid.insert_thing(Simple, grid.start.location)
        vehicle_to_be_tracked.set_weights(w_wall, w_mush, w_cact, w_dest)
        grid.set_tracker(vehicle_to_be_tracked)

        try:
            while not grid.destination_reached(1000) and mass > 0:
                grid.next_turn()
                mass = grid.tracked.mass

            # while

        finally:
            time.sleep(2)
            pygame.quit()

        # try..except

        reached = 0
        if grid.destination_reached():
            reached = 1
            logger.info(f'Destination reached in {grid.turns} turns.')
            
        return reached, grid.turns
        
    ### loop_one_grid ###

    random.seed(42)

    score = pd.DataFrame(index = range(n), columns = ('Wall', 'Mushroom', 
        'Cactus', 'Destination', 'Reached', 'Turns'))

    for i in range(n):
        w_wall: float = 2 * random.random() - 1
        w_mush: float = 2 * random.random() - 1
        w_cact: float = 2 * random.random() - 1
        w_dest: float = 2 * random.random() - 1

        reached, turns = loop_one_grid(w_wall, w_mush, w_cact, w_dest)

        score.loc[i, 'Wall'] = w_wall
        score.loc[i, 'Mushroom'] = w_mush
        score.loc[i, 'Cactus'] = w_cact
        score.loc[i, 'Destination'] = w_dest
        score.loc[i, 'Reached'] = reached
        score.loc[i, 'Turns'] = turns

        logger.info(f'loop {i}: {turns}')

    # for

    logger.info(str(score))

    return

### test_many_vehicles ###


def test_ga(res_path: str, icon_style: int):
    winners = analyse_simple_vehicle(None, None, res_path, 1)
    ga = winners.population[0]
    w_wall = ga.get_var('w_wall')
    w_mush = ga.get_var('w_mushroom')
    w_cact = ga.get_var('w_cactus')
    w_dest = ga.get_var('w_target')

    test_move_auto(res_path, 1, FixedGenerator, w_wall, w_mush, w_cact,w_dest)

    return

### test_ga ###


if __name__ == "__main__":
    res_path='/media/i-files/home/arnold/development/python/ml/vehicles'

    #test_move_around(res_path, 1)
    # test_move_auto(res_path, 1, FixedGenerator, -0.5, 0.5, -1.0, 0.5)
    #test_many_vehicles(res_path, 1, FixedGenerator, 10)
    test_ga(res_path, 1)
   