# import logging
from telnetlib import GA
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
from grid_objects import Wall, Vehicle, Mushroom, Cactus, Rock, Start, Destination, DotGreen
from grid_vehicles import Simple
from grid_generators import RandomGenerator, FixedGenerator

from grid_ga import analyse_simple_vehicle

# Initialize Pandas  display options such that the whole DataFrame is printed
pd.options.display.max_rows = 999999
pd.options.display.max_columns = 999999


def test_move_around(res_path: str, icon_style: int, generator):
    screen_width = 700
    screen_height = 700
    rows = 15
    cols = 21

    # create a generator for this test

    # create a grid with appropriate number of columns and rows
    grid = Grid(generator, 
                grid_size = (cols, rows), 
                res_path = res_path, 
                icon_style = icon_style
               )


    # define a grid viewer for the grid
    grid_viewer = GridView2D(grid, grid.definitions, screen_size=(screen_width, screen_height))
    
    logger.info(grid.print_grid(grid.grid_cells))

    # set vehicle on the start position and have it tracked by the grid
    vehicle_to_be_tracked = grid.insert_thing(Simple, grid.start.location)
    vehicle_to_be_tracked.leave_trace = True
    grid.set_tracker(vehicle_to_be_tracked)
        
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
        time.sleep(60)
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
    grid = Grid(generator, 
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

        # try..finally

        reached = 0
        if grid.destination_reached():
            reached = 1
            logger.info(f'Destination reached in {grid.turns} turns.')
            
        return reached, grid.turns
        
    ### loop_one_grid ###

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
    levels = [3]
    turns = [25]
    max_keep = 5
    initial_pop = []
    for idx in range(len(levels)):
        level = levels[idx]

        # the more difficult the level, the more ietrations
        iters = turns[idx]
        logger.info('')
        logger.info(f'* * * Level {level} with {iters} generations * * *')

        # get a population of GA's that performs the task well
        winners = analyse_simple_vehicle(None, None, 
                                         res_path, 
                                         icon_style = 1, 
                                         level = level, 
                                         initial_pop = initial_pop,
                                         iterations = iters,
                                        )

        # show performance of winning GA
        ga = winners.population[0]
        ga.show()

        w_wall = ga.get_var('w_wall')
        w_mush = ga.get_var('w_mushroom')
        w_cact = ga.get_var('w_cactus')
        w_dest = ga.get_var('w_target')

        test_move_auto(res_path, 1, FixedGenerator(level = level), w_wall, w_mush, w_cact,w_dest)

        # create an initial population to seed the next level with
        initial_pop = []
        for idx in range(max_keep):
            ga: GA = winners.population[idx]
            ga.show()
            
            # check if the GA reached the destination
            reached = ga.fitness_scores['reached']

            # if not, break the loop
            if reached == 0:
                break

            # else append GA to initial_pop
            else:
                initial_pop.append(ga)

            # if
        # for

        if len(initial_pop) > 0:
            logger.info(f'Initial population of {len(initial_pop)} vehicles')

        # if
    # for

    return

### test_ga ###


if __name__ == "__main__":
    random.seed(42)

    res_path='/media/i-files/home/arnold/development/python/ml/vehicles'

    r_generator = RandomGenerator(n_mushrooms=5, n_cactuses=4, n_rocks=3)
    f_generator = FixedGenerator(level = 4)
    test_move_around(res_path, 1, f_generator)
    # test_move_auto(res_path, 1, FixedGenerator, -0.5, 0.5, -1.0, 0.5)
    #test_many_vehicles(res_path, 1, FixedGenerator, 10)
    #test_ga(res_path, 1)
   