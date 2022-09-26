# Code initialisatie: logging
import logging
logger = logging.getLogger()

import sys
import random
import numpy as np
import pandas as pd

from math import sqrt

from grid import Grid
from grid_thing import Thing
from grid_thing_data import COMPASS
from grid_objects import Vehicle
from grid_sensors import Eye
from grid_thing_data import COL_CATEGORY, COL_ENERGY

class Simple(Vehicle):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)
        
        self.type = 'Vehicle'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.direction = 'X'

        # create basic sensors
        self.sensors = [
                        Eye(self, grid, definitions.loc['Wall', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['Mushroom', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['Cactus', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['Destination', COL_CATEGORY]),
                       ]
        
        return
    
    ### __init__ ###

    def set_weights(self,w_wall: float, w_mush: float, w_cact: float, w_dest: float) -> None:
        # weights for each category, default zero
        self.weights = {}
        for cat in self.definitions[COL_CATEGORY].items():
            self.weights[cat] = 0

        # assign specific weights for this vehicle
        self.weights[self.grid.definitions.loc['Wall', COL_CATEGORY]] = w_wall
        self.weights[self.grid.definitions.loc['Mushroom', COL_CATEGORY]] = w_mush
        self.weights[self.grid.definitions.loc['Cactus', COL_CATEGORY]] = w_cact
        self.weights[self.grid.definitions.loc['Destination', COL_CATEGORY]] = w_dest

        return

    ### set_weights ###
    
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

        logger.debug(str(evaluated_moves))

        return max_move

    ### evaluate ###

## Class: Simple ##
