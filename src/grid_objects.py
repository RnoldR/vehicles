# Code initialisatie: logging
import logging
logger = logging.getLogger()

import random
import numpy as np
import pandas as pd

from grid import Grid
from grid_thing import Thing

from grid_thing_data import COMPASS, COL_CATEGORY, COL_MASS, COL_ICON

class Wall(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__('Wall', location, definitions, grid)
        
        return

### Class: Wall ###

class Vehicle(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__('Vehicle', location, definitions, grid)
        
        self.direction = 'X'
        self.leave_trace = False
        
        return
    
    ### __init__ ###
    
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
            if self.Verbose > 0:
                logger.info('Destination reached!')
                    
        # Vehicle may move over the field
        elif idx == self.definitions.loc['Field', COL_CATEGORY]:
            new_loc = potential_loc
            
        # Vehicle may not move thru a wall
        elif idx == self.definitions.loc['Wall', COL_CATEGORY]:
            new_loc = self.location
            if self.Verbose > 0:
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
            if self.Verbose > 0:
                logger.info('Vehicle mass from Mushroom: ' + str(cost))
            
        # Cannot be moved over a cactus which remainslost
        elif idx == self.definitions.loc['Cactus', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            new_loc = self.location
            if self.Verbose > 0:
                logger.info('Vehicle cost from Cactus: ' + str(cost))
            
        # Rock can move, depending on the object before it
        elif idx == self.definitions.loc['Rock', COL_CATEGORY]:
            new_loc = self.location
            if may_move == 'yes' or may_move == 'maybe':
                thing = grid.find_thing_by_loc(potential_loc)
                thing.move(grid, direction)
                new_loc = potential_loc
                
            if self.Verbose > 0:
                logger.info('Vehicle cost from Rock: ' + str(cost))

        # Can move over a green dot which is lost
        elif idx == self.definitions.loc['DotGreen', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc, 'DotGreen')
            thing.deleted = True
            new_loc = potential_loc # self.location
            if self.Verbose > 0:
                logger.info('Vehicle mass from green dot: ' + str(cost))
            
        # Can move over a red dot which is lost
        elif idx == self.definitions.loc['DotRed', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.deleted = True
            new_loc = potential_loc # self.location
            if self.Verbose > 0:
                logger.info('Vehicle mass from red dot: ' + str(cost))
            
        else:
            message = '*** Unknown field code in Rock.move: ' + str(idx)
            logger.critical(message)
            raise ValueError('*** Unknown field code in Rock.move: ' + str(idx))
                
        # if
    
        self.mass += cost
        grid.grid_cells[self.location] = self.definitions.loc['Field', COL_CATEGORY]
        old_loc = self.location
        self.location = new_loc
        grid.grid_cells[self.location] = self.definitions.loc['Vehicle', COL_CATEGORY]
        
        if self.leave_trace:
            grid.add_thing(DotGreen, old_loc)
        
        return cost, self.location
    
    ### move ###
            
## Class: Vehicle ##

class Mushroom(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__('Mushroom', location, definitions, grid)

        return
    
    ### __init__ ###
    
### Class: Mushroom ###

class Cactus(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__('Cactus', location, definitions, grid)

        return
    
    ### __init__ ###

### Class: Cactus ###

class Rock(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__('Rock', location, definitions, grid)

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
            if self.Verbose > 0:
                logger.info(grid.print_grid(grid.grid_cells))
            
        # Rock can move, depending on the object before it
        elif idx == self.definitions.loc['Rock', COL_CATEGORY]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.move(grid, direction)
            new_loc = potential_loc
            
        # if
        if not thing is None and self.Verbose > 0:
            logger.info('Rock added cost from ' + str(thing.type) + 
                        ' cost = ' + str(cost))
    
        grid.grid_cells[self.location] = self.definitions.loc['Field', COL_CATEGORY]
        self.location = new_loc
        grid.grid_cells[self.location] = self.definitions.loc['Rock', COL_CATEGORY]
            
        return cost, self.location
            
## Class: Rock ##
        
class Start(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__('Start', location, definitions, grid)

        self.visible = False
        self.mass = self.max_mass
        
        return
    
    # __init__ #
    
## Class: Start ##

class Destination(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__('Destination', location, definitions, grid)

        self.visible = False
        self.mass = self.max_mass
        
        return
    
    # __init__ #
    
## Class: Destination ##

class DotGreen(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__('DotGreen', location, definitions, grid)

        self.visible = False
        
        return
    
    # __init__ #
    
## Class: DotGreen ##

class DotRed(Thing):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__('DotRed', location, definitions, grid)

        self.visible = False
        
        return
    
    # __init__ #
    
## Class: DotRed ##