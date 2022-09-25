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

#from grid_objects import Start, Destination, Dot_green

from grid_thing_data import ICON_STYLE, COL_CATEGORY, COL_ENERGY, COL_ICON, COL_CLASS

# choose icon style (1-3)    
ICON_STYLE = 1

# Vehicle directions
COMPASS = {"N": (0, -1),
           "E": (1, 0),
           "S": (0, 1),
           "W": (-1, 0),
           "X": (0, 0)
          }

# forward declarations of classes
class Thing: pass
class GridGenerator: pass

class Grid:
    def __init__(self, generator=None, grid_size=(10,10), definitions=None):
        # Assign parameters
        self.definitions = definitions
        
        # grid's configuration parameters
        if not (isinstance(grid_size, (list, tuple)) and len(grid_size) == 2):
            raise ValueError("grid_size must be a tuple: (width, height).")
        
        self.grid_size = grid_size
        self.turns: int = 0
        self.start = None
        self.destination = None

        # grid member variables
        self.things_by_id = {}
        self.vehicles_by_id = {}
        self.tracked = None

        # list of all cell locations
        self.grid_cells = np.zeros(self.grid_size, dtype=int)
        
        # generate the grid when a generator is present
        if generator is not None:
            generator.generate(self)
            
        return

    def save_grid(self, file_path):
        if not isinstance(file_path, str):
            logging.critical("Invalid file_path. It must be a string.")
            return False

        elif not os.path.exists(os.path.dirname(file_path)):
            logging.critical("Cannot find the directory for " + file_path)
            return False

        else:
            np.save(file_path, self.grid_cells, allow_pickle=False, fix_imports=True)
            return True

    @classmethod
    def load_grid(cls, file_path):
        if not isinstance(file_path, str):
            logging.critical("Invalid file_path. It must be a string: " + file_path)
            return None

        elif not os.path.exists(file_path):
            logging.critical("Cannot find grid file: " + file_path)
            return None

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)
        
    def generate_grid(self, generator: GridGenerator) -> None:
        generator.generate(self)
         
        return

    def print_grid(self, matrix) -> None:
        cols, rows = matrix.shape
        strmat = 'Grid size (width x height): {:d} x {:d}\n'.format(cols, rows)
        
        for row in range(rows):
            line = ''
            for col in range(cols):
                line += '{:2d}'.format(matrix[col, row])
            # for
            strmat += line + '\n'
        # for
        
        return strmat
    
    def insert_thing(self, ThingClass, loc) -> Thing:
        thing = ThingClass(loc, self.definitions, self)
        self.grid_cells[loc] = self.definitions.loc[thing.type, COL_CATEGORY]
        self.things_by_id[thing.id] = thing
        
        return thing
    
    ### insert_thing ###
    
    def insert_things(self, ThingClass, locs) -> list:
        things = []
        if not locs is None:            
            for loc in locs:
                thing = self.insert_thing(ThingClass, loc)
                things.append(thing)
        
        return things

    ### insert_things ###
    
    def set_tracker(self, thing: Thing) -> None:
        self.tracked = thing
        
        return

    ### set_tracker ###

    def set_start(self, ThingClass, loc: tuple) -> None:
        self.start = self.insert_thing(ThingClass, loc)
        
        return

    ### set_start ###
    
    def set_destination(self, ThingClass, loc: tuple) -> None:
        self.destination = self.insert_thing(ThingClass, loc)
        
        return

    ### set_destination ###
    
    def process_command(self, command: str, grid_pos: tuple, 
                        definitions: pd.DataFrame) -> None:
        
        if command == 'P':
            logger.info(self.print_grid(self.grid_cells))
        elif command == '-':
            self.find_route()
        elif command in ['m', 'c', 'r', 'v', 'w']:
            if self.grid.grid_cells[grid_pos] != 0:
                logger.warning('Already occupied ' + str(grid_pos))
            else:
                inserter = definitions[definitions['Command'] == command]
                if len(inserter) != 1:
                    logger.warning('Wrong number of entries{:d} for {:s}'
                                   .format(len(inserter), command))
                else:
                    logger.debug(str(inserter))
                    class_def = inserter['Class']
                    
                    thing = self.grid.insert_thing(class_def, grid_pos)

                    logger.info('Inserted ' + str(thing.type) + ' at ' +
                                str(thing.location))

                # if
            # if
                
        elif command == 'f':
            thing = self.grid.find_thing_by_loc(grid_pos)
            self.grid.remove_thing(thing)
            if thing is None:
                logger.info('Nothing found')
            else:
                logger.info('Removed ' + str(thing.type) + ' at ' +
                            str(thing.location))
            # if
        
        # if

        return

    ### process_command###
    
    def generate_random_locs(self, num_things):
        # Check if things should be generated
        if num_things <= 0:
            return
        
        # Generate all empty cells
        cell_ids = [(x, y) for y in range(self.grid_size[1]) 
                            for x in range(self.grid_size[0]) 
                                if self.grid_cells[x, y] == 0]
    
        # limit the maximum number of things to half the number of cells available.
        max_things = int(self.grid_size[0] * self.grid_size[1] / 2)
        num_things = min(max_things, num_things)
        thing_locations = random.sample(cell_ids, num_things)

        return thing_locations
    
    ### generate_random_loc ###
    
    def find_thing_by_loc(self, loc, type=''):
        for key in self.things_by_id.keys():
            thing = self.things_by_id[key]
            if loc == thing.location:
                if type == '':
                    return thing
                else:
                    if thing.type == type:
                        return thing
        
        return None
    
    ### find_thing_by_loc ###
    
    def find_thing_by_type(self, thing_type: str):
        for key in self.things_by_id.keys():
            thing = self.things_by_id[key]
            if thing.type == thing_type:
                return thing
                
        return None

    ### find_thing_by_type ###

    def find_category_at_loc(self, loc: tuple):
        cat = self.grid_cells[loc]

        return cat
    
    ### find_category_at_loc ###

    def remove_thing(self, thing):
        if thing is None:
            logger.warning('Grid.remove_thing: argument is None')
        else:
            id = thing.id
            if id in self.things_by_id.keys():
                if thing.category == self.grid_cells[thing.location]:
                    self.grid_cells[self.things_by_id[id].location] = \
                        self.definitions.loc['Field', COL_CATEGORY]
                        
                del self.things_by_id[id]
                logger.info(str(thing.type) + ' removed: ' + str(id))
            else:
                logger.warning('No ' + str(thing.type) + ' found: ' + str(id))
            
        return

    ### remove_thing ###
    
    def move_things(self):
        # Move all things
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            thing.move(self)
            
        # remove things being flagged as deleted
        removes = []
        for id in self.things_by_id:
            if self.things_by_id[id].deleted:
                removes.append(id)
            
        # Kill removed things
        for id in removes:
            thing = self.things_by_id[id]
            self.remove_thing(thing)

        return
    
    ### move_things ###
    
    def next_turn(self):
        self.turns += 1

        # Move all things
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            thing.next_turn()
            
        # remove things being flagged as deleted
        removes = []
        for id in self.things_by_id:
            if self.things_by_id[id].deleted:
                removes.append(id)
            
        # Kill removed things
        for id in removes:
            thing = self.things_by_id[id]
            self.remove_thing(thing)

        return

    ### next_turn ###

    def destination_reached(self, max_turns: int = 0) -> bool:

        if max_turns > 0 and self.turns >= max_turns:
            return True

        elif self.tracked.location[0] == self.destination.location[0] and \
           self.tracked.location[1] == self.destination.location[1]:

            return True

        else:
            return False

    ### destination_reached ###
    
    def get_n_things(self, type_str):
        n = 0
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            if thing.type == type_str:
                n += 1
                
        return n

    ### get_n_things ###
       
    def get_vehicles_energy(self, ThingClass):
        energy = 0
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            if isinstance(thing, ThingClass):
                energy += thing.energy
                
        return energy

    ### get_vehicles_energy ###
    
    @staticmethod    
    def load_config(filename: str):
        """
        Load configuration from yaml file

        filename (str): name of configuration file
        """

        with open(filename) as yaml_data:
            config = yaml.load(yaml_data, Loader=yaml.FullLoader)
        
        for key, value in config['Things'].items():
            logger.debug(str(key) + ': ID = {:d}, cost = {:.2f}, growth = {:.2f}'.
                         format(value[0], value[1], value[2]))
            
        return config['Things']
    
    # load_config #

    @staticmethod    
    def load_resources(resources, res_path: str, style: int=1): 
        for key in resources:
            filename = key.lower() + '-' + str(style) + '.png'
            filename = os.path.join(res_path, filename)
            
            img = pygame.image.load(filename)
            
            resources[key].append(img)
            
        df = pd.DataFrame.from_dict(resources, orient='index',
                                    columns=[COL_CATEGORY, 'Cost', 'Growth', 'Command', 'Image'])
        
        return df
    
    # load_resources #
    
    def make_step(self, a: np.array, m: np.array, k: int):
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i][j] == k:
                    if i>0 and m[i-1][j] == 0 and \
                        (a[i-1][j] == 0 or 
                         a[i-1][j] == self.definitions.loc['Destination', COL_CATEGORY]):
                        m[i-1][j] = k + 1
                    if j>0 and m[i][j-1] == 0 and \
                        (a[i][j-1] == 0 or
                         a[i][j-1] == self.definitions.loc['Destination', COL_CATEGORY]):
                        m[i][j-1] = k + 1
                    if i<len(m)-1 and m[i+1][j] == 0 and \
                        (a[i+1][j] == 0 or
                         a[i+1][j] == self.definitions.loc['Destination', COL_CATEGORY]):
                        m[i+1][j] = k + 1
                    if j<len(m[i])-1 and m[i][j+1] == 0 and \
                        (a[i][j+1] == 0 or
                         a[i][j+1] == self.definitions.loc['Destination', COL_CATEGORY]):
                        m[i][j+1] = k + 1
                    # if
                # if
            # for
        # for
                        
        return

    ### make_step ###
           
    def find_route(self):
        # find route from vehicle to its destination
        vehicle = self.find_thing_by_type('Vehicle')
        destination = self.find_thing_by_type('Destination')
        
        # both must exist
        if vehicle is None or destination is None:
            logger.critical('*** find_route: vehicle or destination is None')
            return
        
        # find start and end location 
        start = vehicle.location
        end = destination.location
        
        # Create an empty m array of sketching and testing the route
        m = np.zeros(self.grid_cells.shape, dtype=np.int)
        k = 0
        m[start] = 1

        while k < 32:# m[end] == 0:
            k += 1
            self.make_step(self.grid_cells, m, k)

        # while
        
        logger.info(self.print_grid(m))
        i, j = end
        k = m[i][j]
        the_path = [(i,j)]
        while k > 1:
            if i > 0 and m[i - 1][j] == k-1:
                i, j = i-1, j
                the_path.append((i, j))
                k-=1
            elif j > 0 and m[i][j - 1] == k-1:
                i, j = i, j-1
                the_path.append((i, j))
                k-=1
            elif i < len(m) - 1 and m[i + 1][j] == k-1:
                i, j = i+1, j
                the_path.append((i, j))
                k-=1
            elif j < len(m[i]) - 1 and m[i][j + 1] == k-1:
                i, j = i, j+1
                the_path.append((i, j))
                k -= 1
            # if
        # while
        the_path.reverse()
        
        return  the_path

    ### find_route ###
        
## Class: Grid ##

class GridGenerator(object):
    def __init__(self):
        return
    
    def generate(self, grid: Grid):
        return
    
