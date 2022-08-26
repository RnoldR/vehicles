#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:25:15 2021

@author: arnold
"""

import os
import time
import yaml
import pygame
import random
import numpy as np
import pandas as pd

from math import sqrt

from grid import Grid, COMPASS

# Code initialisatie: logging
import logging
import importlib
importlib.reload(logging)

# create logger
logger = logging.getLogger('Grid2D')

logger.setLevel(10)

# create file handler which logs even debug messages
fh = logging.FileHandler('grid-view-2D.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(logging.Formatter('%(message)s'))

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# forward declaration
class Thing: pass

class Sensor():
    def __init__(self, owner: Thing, world: Grid):
        self.owner = owner
        self.world = world
        
        return
    
    ### __init__ ###
    
### Class: Sensor ###
        
class Thing():
    # Define static sequence number to have unique ID's
    Seq: int = 0
    DefaultType: str = 'Field'
    DefaultCategory: int = 0
    DefaultMass: float = 50
    MaxMass: float = 100
    
    def __init__(self, location: tuple, definitions: pd.DataFrame):
        # increment sequence number
        Thing.Seq += 1

        # system attributes
        self.id: int = Thing.Seq
        self.location: tuple = location
        self.definitions: pd.DataFrame = definitions
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
        
        if idx == self.definitions.loc['Field']['ID']:
            cost = self.definitions.loc['Field']['Cost']
            may_move = 'yes'
        elif idx == self.definitions.loc['Wall']['ID']:
            cost = self.definitions.loc['Wall']['Cost']
            may_move = 'no'
        elif idx == self.definitions.loc['Vehicle']['ID']:
            cost = thing.energy
            may_move = 'no'
        elif idx == self.definitions.loc['Mushroom']['ID']:
            cost = thing.energy
            may_move = 'maybe'
        elif idx == self.definitions.loc['Cactus']['ID']:
            cost = thing.energy
            may_move = 'maybe'
        elif idx == self.definitions.loc['Rock']['ID']:
            cost, may_move = thing.cost(grid, direction)
            cost = -abs(cost) + thing.energy
        elif idx == self.definitions.loc['Start']['ID']:
            cost = self.definitions.loc['Start']['Cost']
            may_move = 'yes'
        elif idx == self.definitions.loc['Destination']['ID']:
            cost = self.definitions.loc['Destination']['Cost']
            may_move = 'yes'
        elif idx == self.definitions.loc['Dot_green']['ID']:
            cost = self.definitions.loc['Dot_green']['Cost']
            may_move = 'yes'
        elif idx == self.definitions.loc['Dot_red']['ID']:
            cost = self.definitions.loc['Dot_red']['Cost']
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
