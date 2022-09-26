# Code initialisatie: logging
import logging
logger = logging.getLogger()

import os
import time
import yaml
import pygame
import random
import numpy as np
import pandas as pd

from math import sqrt

from grid import Grid, COMPASS

from grid_thing_data import ICON_STYLE, COL_CATEGORY, COL_ENERGY, COL_ICON, COL_CLASS

# forward declaration
class Thing: pass
class Grid: pass

class Thing():
    # Define static sequence number to have unique ID's
    Seq: int = 0
    DefaultType: str = 'Field'
    DefaultCategory: int = 0
    DefaultMass: float = 50
    MaxMass: float = 100
    Verbose: int = 1
    
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        # increment sequence number
        Thing.Seq += 1

        # system attributes
        self.id: int = Thing.Seq
        self.location: tuple = location
        self.definitions: pd.DataFrame = definitions
        self.grid = grid
        self.deleted: bool = False
        
        # all things are field by default
        self.type: str = Thing.DefaultType
        self.category: int = Thing.DefaultCategory
        
        # some general attributes
        self.visible: bool = True
        self.age: int = 0
        self.mass: float = Thing.DefaultMass
        self.color = {'r': 0, 'g': 0, 'b': 0}
        self.energy: float = 0
        self.sensors = []
        self.effectors = []

        return
    
    ### __init__ ###
    
    def d(self, loc) -> float:
        if self.location != None:
            dx = self.location[0] - loc[0]
            dy = self.location[1] - loc[1]
            
            dd = dx * dx + dy * dy
            
            if dd > 0:
                return sqrt(dd)
            else:
                return 0
        else:
            return -1
        
    ### d ###
    
    def cost(self, grid, direction):
        ''' Computes the cost of a move in a certain direction
        
        A rock may move when it is pushed by a vehicle, depending on the 
        field lying before it. There is always a cost involved, even when 
        the rock cannot move. When a rock is pushed against a wall but the
        cost that is transmitted to the vehicle is the energy of the rock
        plus the nerge of the wall. For a cactus the same type of cost 
        is computed, but the rock will move over the cactus and the cactus
        will disappear. Thus this is a save way to get rid of cactuses.
        
        Cost is always negative.

        Args:
            grid (np.arry): grid on which to perform the move
            direction (char): direction in which to move
            
        Returns:
            The cost of the move when it should be effected (float)
        '''
        potential_pos = (self.location[0] + COMPASS[direction][0], self.location[1] + COMPASS[direction][1])
        idx = grid.grid_cells[potential_pos]
        thing = grid.find_thing_by_loc(potential_pos)
        mess = 'None' if thing is None else thing.type
        logger.debug('{:s} - {:s} -> {:s} idx = {:d} thing.type = {:s}'.
                     format(str(self.type), str(self.location), str(potential_pos), idx, mess))
        cost = 0
        may_move = 'no'
        
        if idx == self.definitions.loc['Field', COL_CATEGORY]:
            cost = self.definitions.loc['Field', COL_ENERGY]
            may_move = 'yes'
        elif idx == self.definitions.loc['Wall', COL_CATEGORY]:
            cost = self.definitions.loc['Wall', COL_ENERGY]
            may_move = 'no'
        elif idx == self.definitions.loc['Vehicle', COL_CATEGORY]:
            cost = thing.energy
            may_move = 'no'
        elif idx == self.definitions.loc['Mushroom', COL_CATEGORY]:
            cost = thing.energy
            may_move = 'maybe'
        elif idx == self.definitions.loc['Cactus', COL_CATEGORY]:
            cost = thing.energy
            may_move = 'maybe'
        elif idx == self.definitions.loc['Rock', COL_CATEGORY]:
            cost, may_move = thing.cost(grid, direction)
            cost = -abs(cost) + thing.energy
        elif idx == self.definitions.loc['Start', COL_CATEGORY]:
            cost = self.definitions.loc['Start', COL_ENERGY]
            may_move = 'yes'
        elif idx == self.definitions.loc['Destination', COL_CATEGORY]:
            cost = self.definitions.loc['Destination', COL_ENERGY]
            may_move = 'yes'
        elif idx == self.definitions.loc['Dot_green', COL_CATEGORY]:
            cost = self.definitions.loc['Dot_green', COL_ENERGY]
            may_move = 'yes'
        elif idx == self.definitions.loc['Dot_red', COL_CATEGORY]:
            cost = self.definitions.loc['Dot_red', COL_ENERGY]
            may_move = 'yes'
        else:
            raise ValueError('*** Unknown field code in Rock.move:', idx)
            
        return cost, may_move
    
    ### cost ###
    
    def next_turn(self):
        self.age += 1

        return
    ### next_turn ###
    
    def move(self, grid, direction=None):
        
        return
    
    ### move ###
    
### Class: Thing ###
