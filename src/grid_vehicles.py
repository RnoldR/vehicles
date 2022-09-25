#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:17:21 2021

@author: arnold
"""

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

class Eye(Sensor):
    def __init__(self, owner: Thing, grid: Grid, sensitivity: int):
        super().__init__(owner, grid)

        self.sensitive_for: int = sensitivity
        
        return
    
    ### __init__  ###

    @staticmethod    
    def sensor_value(self, signal: float, x: int, y: int, id: int):

        return (signal, x, y, id)

    ### sensor_value ###
    
    def sense_objects(self, grid: Grid) -> dict:
        """
        Senses a square of the grid. The square is a dictionary with four
        keys: (lower x, lower y, upper x, upper y). Only grid elements
        within this square will be sensed.

        Parameters
        ----------
        grid : Grid
            The grid to be sensed.
        square : tuple
            dictionary with keys (lower x, lower y, upper x, upper y).
        loc: (tuple)
            Position of the sensor: tuple (x, y)

        Returns
        -------
        A dictionary containg the normalized colors of all objects as rgb values.

        """

        # pre-select the objects this sensor is sensitive for        
        things = [grid.things_by_id[k] for k in grid.things_by_id 
                  if grid.things_by_id[k].category == self.sensitive_for]

        perceptions = []
        for thing in things:
            if thing.category == self.sensitive_for:
                # signal is mass
                signal = thing.mass

                # normalize mass by diving by max mass   
                norm_signal = signal / Thing.MaxMass

                # add to total perceptions
                perceptions.append((norm_signal, thing.location[0], thing.location[1], thing.id))     

            # if

        # for  

        # sort by normalized signal strength in descending order
        if len (perceptions) > 0:
            perceptions = sorted(perceptions, key=lambda tup: tup[0], reverse = True)

        # if
        
        return perceptions
    
    ### sense ###
    
class Simple(Vehicle):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)
        
        self.type = 'Vehicle'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.direction = 'X'

        # weights for each category, default zero
        self.weights = {}
        for cat in self.definitions[COL_CATEGORY].items():
            self.weights[cat] = 0

        # assign specific weights for this vehicle
        self.weights[definitions.loc['Wall', COL_CATEGORY]] = -0.75
        self.weights[definitions.loc['Mushroom', COL_CATEGORY]] = 0.50
        self.weights[definitions.loc['Cactus', COL_CATEGORY]] = -1.0
        self.weights[definitions.loc['Destination', COL_CATEGORY]] = 0.5

        # create basic sensors
        self.sensors = [
                        Eye(self, grid, definitions.loc['Wall', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['Mushroom', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['Cactus', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['Destination', COL_CATEGORY]),
                       ]
        
        return
    
    ### __init__ ###
    
    def next_turn(self):
        super().next_turn()
        
        perceptions = self.perceive()
        self.direction = self.evaluate(perceptions)
        self.move(self.grid)

        return
    
    ### next_turn ###

    def perceive(self):
        # get the stronfest signal for all sensors
        perceptions = {}

        # enumerate over all (type of) sensors
        for sensor in self.sensors:
            # get their results
            sensed = sensor.sense_objects(self.grid)

            # if there are results, add the first one to the perception category
            perceptions[sensor.sensitive_for] = sensed

        # for

        # dictionary now contains for each category a list of perceptions 
        # ordered by descending signal strength, return it.

        return perceptions

    ### perceive ###

    def evaluate(self, perceptions: dict):
        """ evaluates a next move based on perceptions of the environment

        strategy for this vehicle
          1. move in the direction of the destination
          2. avoid cactuses at all cost
          3. permit a small detour to eat a mushroom

        Args:
            perceptions (dict): for each category a list of perceptions 
                ordered by descending signal strength

        Returns:
            _type_: advised move
        """

        (x, y) = self.location
        possible_moves = {
                          'N': (x, y - 1),
                          'E': (x + 1, y), 
                          'S': (x, y + 1), 
                          'W': (x - 1, y), 
                         }

        evaluated_moves = {}
        max_val = -1_000_000
        max_move = None

        # evaluate each possible move
        for move in possible_moves:
            energy = 0

            # look for all perceived objects
            for cat in perceptions:

                # if there is an object perceived
                if len(perceptions[cat]) > 0:
                    # get the first object (with the most signal strength)
                    perception = perceptions[cat][0]

                    # compute its distance to this possible move
                    val = possible_moves[move]
                    d = (val[0] - perception[1]) ** 2 + (val[1] - perception[2]) ** 2
                    if d > 0:
                        d = sqrt(d)
                    else:
                        d = 0

                    # divide signal strength by distance
                    signal_strength = perception[0]
                    if d > 0:
                        signal_strength /= d

                    # and by weight of this category
                    weighted_signal = self.weights[cat] * signal_strength

                    # add to energy
                    energy += weighted_signal

                    logger.debug(cat, move, val, perception, d, signal_strength, weighted_signal, energy)

                # if
            # for

            evaluated_moves[move] = energy

            if energy > max_val:
                max_val = energy
                max_move = move

        # for

        logger.info(str(evaluated_moves))

        # sys.exit()
        return max_move

    ### evaluate ###
    
    def move(self, grid):
        direction = self.direction
        self.direction = "X"
        
        if direction == "X":
            direction = random.sample(['N', 'E', 'S', 'W'], 1)[0]
            
        potential_loc = (self.location[0] + COMPASS[direction][0], self.location[1] + COMPASS[direction][1])
        idx = grid.grid_cells[potential_loc]
        cost, may_move = self.cost(grid, direction)
        
        # Vehicle may have reached destination
        if idx == self.definitions.loc['Destination', COL_CATEGORY]:
            new_loc = potential_loc
            logger.info('!!!Destination reached!!!')
                    
        # Vehicle may move over the field
        elif idx == self.definitions.loc['Field', COL_CATEGORY]:
            new_loc = potential_loc
            
        # Vehicle may not move thru a wall
        elif idx == self.definitions.loc['Wall', COL_CATEGORY]:
            new_loc = self.location
            logger.info('Vehicle cost from Wall: ' + str(cost))
            
        # Rock cannot be pushed thru a Vehicle
        elif idx == self.definitions.loc['Vehicle', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            new_loc = self.location
            
        # Can move over a mushroom which is lost
        elif idx == self.definitions.loc['Mushroom', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.deleted = True
            new_loc = potential_loc # self.location
            logger.info('Vehicle energy from Mushroom: ' + str(cost))
            
        # Cannot be moved over a cactus which remainslost
        elif idx == self.definitions.loc['Cactus', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            new_loc = self.location
            logger.info('Vehicle cost from Cactus: ' + str(cost))
            
        # Rock can move, depending on the object before it
        elif idx == self.definitions.loc['Rock', COL_CATEGORY]:
            new_loc = self.location
            if may_move == 'yes' or may_move == 'maybe':
                thing = grid.find_thing_by_loc(potential_loc)
                thing.move(grid, direction)
                new_loc = potential_loc
                
            logger.info('Vehicle cost from Rock: ' + str(cost))

        # Can move over a green dot which is lost
        elif idx == self.definitions.loc['Dot_green', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc, 'Dot_green')
            thing.deleted = True
            new_loc = potential_loc # self.location
            logger.info('Vehicle energy from green dot: ' + str(cost))
            
        # Can move over a red dot which is lost
        elif idx == self.definitions.loc['Dot_red', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.deleted = True
            new_loc = potential_loc # self.location
            logger.info('Vehicle energy from red dot: ' + str(cost))
            
        else:
            message = '*** Unknown field code in Rock.move: ' + str(idx)
            logger.critical(message)
            raise ValueError('*** Unknown field code in Rock.move: ' + str(idx))
                
        # if
    
        self.energy += cost
        grid.grid_cells[self.location] = self.definitions.loc['Field', COL_CATEGORY]
        self.location = new_loc
        grid.grid_cells[self.location] = self.definitions.loc['Vehicle', COL_CATEGORY]
        
        return cost, self.location
    
    ### move ###
            
## Class: Simple ##
