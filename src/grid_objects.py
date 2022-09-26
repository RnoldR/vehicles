#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:17:21 2021

@author: arnold
"""

import random
import numpy as np
import pandas as pd

from grid import Grid, COMPASS
from grid_thing import Thing

from grid_thing_data import ICON_STYLE, COL_CATEGORY, COL_ENERGY, COL_ICON, COL_CLASS

# Code initialisatie: logging
import logging
logger = logging.getLogger()

"""
class Eye(Sensor):
    def __init__(self, category: int):
        super().__init__()

        self.sensitive_for = category
        
        return
    
    ### __init__  ###
    
    def sense_objects(self, grid: Grid, square: dict) -> dict:
        
        perceptions = []
        for id, thing in grid.things_by_id:
            if thing.category == self.sensitive_for:
                # compute distance to perceived object
                d = thing.d(self.owner.location)

                # compute signal
                signal = d * thing.mass 

                # cnormalize signal by diving by max mass   
                norm_signal = signal / thing.DefaultMass

                # add to total perceptions
                perceptions.append((norm_signal, thing.location[0], thing.location[1], id))     

            # if

        # for  

        # sort by normalized signal strength in descending order
        if len (perceptions) > 0:
            perceptions = sorted(perceptions, key=lambda tup: tup[0], reverse = True)
        
        return perceptions
    
    ### sense ###
    
    def sense_square(self, grid: Grid, square: dict, loc: tuple) -> dict:
        rgb = {'r': 0, 'g': 0, 'b': 0}
        n = 0
        for x in range(square['lower x'], square['upper x']):
            for y in range(square['lower y'], square['upper y']):
                if grid.grid_cells[x, y] != self.definitions.loc['Field', COL_CATEGORY]:
                    # ... haal hier de kleuren op en sommeer ze in rgb
                    # ... vergeet niet n bij te houden om te normaliseren
                    # find 
                    thing = grid.find_thing_by_loc((x, y))
                    d = thing.d(loc)
                    d2 = d * d
                    for channel in thing.color:
                        rgb[channel]  += thing.color[channel] / d2
                        
                    n += 1
        
        n *= 255
        for channel in thing.color:
            rgb[channel]  = [channel] / n
            
        return rgb
    
    ### sense ###
    
### Class: Eye ###
"""
class Wall(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)
        
        self.type = 'Wall'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.color = {'r': 255, 'g': 112, 'b': 67}
        
        return

### Class: Wall ###

class Vehicle(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)
        
        self.type = 'Vehicle'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.direction = 'X'
        self.color = {'r': 0, 'g': 140, 'b': 180}
        
        return
    
    ### __init__ ###
    
    def next_turn(self):
        super().next_turn()
        
        return
    
    ### next_turn ###
    
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
            
## Class: Vehicle ##

class Mushroom(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)

        self.type = 'Mushroom'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.color = {'r': 220, 'g': 80, 'b': 80}
        self.growth = 0.01
        
        return
    
    ### __init__ ###
    
    def next_turn(self):
        super().next_turn()
        
        self.mass += self.growth * self.mass
        if self.mass > Thing.MaxMass:
            self.mass = Thing.MaxMass
            
        return
    ### next_turn ###        

### Class: Mushroom ###

class Cactus(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)

        self.type = 'Cactus'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.color = {'r': 60, 'g': 150, 'b': 20}
        self.growth = 0.01
        
        return
    
    ### __init__ ###

    def next_turn(self):
        super().next_turn()
        
        self.mass += self.growth * self.mass
        if self.mass > Thing.MaxMass:
            self.mass = Thing.MaxMass
            
        return
    ### next_turn ###        

### Class: Cactus ###

class Rock(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)

        self.type = 'Rock'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.color = {'r': 200, 'g': 200, 'b': 200}
        
        return
    
    # __init__ #

    def move(self, grid, direction=None):
        # When direction is None this function is called to move itself, 
        # not from vehicle move (pushing the rock). In that case it returns
        # immediately as it does not spontaneously move
        if direction is None:
            return
            
        # Compute a move based on the push of a vehicle
        potential_loc = (self.location[0] + COMPASS[direction][0], 
                         self.location[1] + COMPASS[direction][1])
        idx = grid.grid_cells[potential_loc]
        cost, new_loc = self.cost(grid, direction)
        thing = None
        
        # Rock may move over a field
        if idx == self.definitions.loc['Field', COL_CATEGORY]:
            new_loc = potential_loc
            
        # Rock may not move thru a wall
        elif idx == self.definitions.loc['Wall', COL_CATEGORY]:
            new_loc = self.location
            
        # Rock cannot be pushed thru a Vehicle
        elif idx == self.definitions.loc['Vehicle', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            new_loc = self.location
            
        # Can be pushed over a mushroom which is lost
        elif idx == self.definitions.loc['Mushroom', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.deleted = True
            new_loc = potential_loc
            
        # Can be pushed over a cactus which is lost
        elif idx == self.definitions.loc['Cactus', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.deleted = True
            new_loc = potential_loc
            grid.grid_cells[new_loc] = self.definitions.loc['Rock', COL_CATEGORY]
            if Thing.Verbose > 0:
                logger.info(grid.print_grid(grid.grid_cells))
            
        # Rock can move, depending on the object before it
        elif idx == self.definitions.loc['Rock', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.move(grid, direction)
            new_loc = potential_loc
            
        # if
        if not thing is None and Thing.Verbose > 0:
            logger.info('Rock added cost from ' + str(thing.type) + 
                        ' cost = ' + str(cost))
    
        grid.grid_cells[self.location] = self.definitions.loc['Field', COL_CATEGORY]
        self.location = new_loc
        grid.grid_cells[self.location] = self.definitions.loc['Rock', COL_CATEGORY]
            
        return cost, self.location
            
## Class: Rock ##
        
class Start(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)

        self.type = 'Start'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.color = {'r': 0, 'g': 255, 'b': 20}
        self.visible = False
        self.mass = self.MaxMass
        
        return
    
    # __init__ #
    
## Class: Start ##

class Destination(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)

        self.type: str = 'Destination'
        self.category: int = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy: float = self.definitions.loc[self.type, COL_ENERGY]
        self.color = {'r': 230, 'g': 0, 'b': 30}
        self.visible = False
        self.mass = self.MaxMass
        
        return
    
    # __init__ #
    
## Class: Destination ##

class Dot_green(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)

        self.type = 'Dot_green'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.color = {'r': 0, 'g': 255, 'b': 0}
        self.visible = False
        
        return
    
    # __init__ #
    
## Class: Dot_green ##

class Dot_red(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)

        self.type = 'Dot_red'
        self.category = self.definitions.loc[self.type, COL_CATEGORY]
        self.energy = self.definitions.loc[self.type, COL_ENERGY]
        self.color = {'r': 255, 'g': 0, 'b': 0}
        self.visible = False
        
        return
    
    # __init__ #
    
## Class: Dot_red ##