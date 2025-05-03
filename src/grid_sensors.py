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

from grid import Grid
from grid_thing import Thing

class Sensor():
    """ Sensor base class. 
    """
    def __init__(self, owner: Thing, grid: Grid):
        """ Each sensor has an owner and a world it can perceive.

        Args:
            owner (Thing): Owner of the sensor
            grid (Grid): The world to be perceived
        """
        
        self.owner = owner
        self.grid = grid
        
        return
    
    ### __init__ ###
    
### Class: Sensor ###
        
class Eye(Sensor):
    """ Sensor able to detect other things. It can detect one thing,
        determined by its category, stored in self.sensitive_for

    Args:
        Sensor (object): root class for Eye
    """

    def __init__(self, owner: Thing, grid: Grid, sensitivity: int):
        super().__init__(owner, grid)

        self.sensitive_for: int = sensitivity
        
        return
    
    ### __init__  ###

    @staticmethod    
    def sensor_value(self, signal: float, x: int, y: int, id: int):

        return (signal, x, y, id)

    ### sensor_value ###
    
    def sense_objects(self) -> dict:
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
        things = [self.grid.things_by_id[k] for k in self.grid.things_by_id 
                  if self.grid.things_by_id[k].category == self.sensitive_for]

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

    ### sense_objects ###

    def sense_layer(self) -> any:
        matrix = np.zeros(self.grid.grid_cells.shape)

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if self.grid.grid_cells[row, col] == self.sensitive_for:
                    matrix[row, col] = self.grid.grid_cells[row, col]

                # if
            # for
        # for

        return matrix
    
    ### sense_layer ###