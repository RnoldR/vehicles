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

        self.set_weights(-0.75, 0.5, -1.0, 0.5)

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
            if Thing.Verbose > 0:
                logger.info('!!!Destination reached!!!')
                    
        # Vehicle may move over the field
        elif idx == self.definitions.loc['Field', COL_CATEGORY]:
            new_loc = potential_loc
            
        # Vehicle may not move thru a wall
        elif idx == self.definitions.loc['Wall', COL_CATEGORY]:
            new_loc = self.location
            if Thing.Verbose > 0:
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
            if Thing.Verbose > 0:
                logger.info('Vehicle energy from Mushroom: ' + str(cost))
            
        # Cannot be moved over a cactus which remainslost
        elif idx == self.definitions.loc['Cactus', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            new_loc = self.location
            if Thing.Verbose > 0:
                logger.info('Vehicle cost from Cactus: ' + str(cost))
            
        # Rock can move, depending on the object before it
        elif idx == self.definitions.loc['Rock', COL_CATEGORY]:
            new_loc = self.location
            if may_move == 'yes' or may_move == 'maybe':
                thing = grid.find_thing_by_loc(potential_loc)
                thing.move(grid, direction)
                new_loc = potential_loc
                
            if Thing.Verbose > 0:
                logger.info('Vehicle cost from Rock: ' + str(cost))

        # Can move over a green dot which is lost
        elif idx == self.definitions.loc['Dot_green', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc, 'Dot_green')
            thing.deleted = True
            new_loc = potential_loc # self.location
            if Thing.Verbose > 0:
                logger.info('Vehicle energy from green dot: ' + str(cost))
            
        # Can move over a red dot which is lost
        elif idx == self.definitions.loc['Dot_red', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.deleted = True
            new_loc = potential_loc # self.location
            if Thing.Verbose > 0:
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
