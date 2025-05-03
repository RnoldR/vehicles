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

from grid_thing_data import COL_CATEGORY


class Grid:
    """ Represnts a 2-dimensional grid. Can be populated by GridThing objects.
    """
    def __init__(
            self, 
            generator = None, 
            grid_size = (8,12), 
            res_path: str = '', 
            icon_style: int = 1,
            verbose: int = 1
        ):
        """ Initializes Grid.

        Args:
            generator (GridGenerator, optional): GridGenerator to populate 
                the Grid. Defaults to None.
            grid_size (tuple, optional): Size of Grid in (Rows, Coluemns). 
                Defaults to (8,12).
            res_path (str, optional): Path to resources. Defaults to ''.
            icon_style (int, optional): Selects type of icons for the GridThing. 
                Defaults to 1.
            verbose (int, optional): Talkative (1) or not (0). Defaults to 1.

        Raises:
            ValueError: The size of the Grid should be a tuple with two integers.
        """
        # Assign parameters
        self.definitions = self.load_thing_definitions(res_path, icon_style)
        
        # grid's configuration parameters
        if not (isinstance(grid_size, (list, tuple)) and len(grid_size) == 2):
            raise ValueError("grid_size must be a tuple: (width, height).")
        
        # Initialize Grid variables
        self.grid_size = grid_size
        self.turns: int = 0
        self.start = None
        self.destination = None
        self.verbose = verbose

        # grid member variables
        self.things_by_id = {}
        self.vehicles_by_id = {}
        self.things_to_be_added_at_end_of_turn = []
        self.tracked = None

        # list of all cell locations
        self.grid_cells = np.zeros(self.grid_size, dtype=int)
        
        # generate the grid when a generator is present
        if generator is not None:
            generator.generate(self)
            
        return

    ### __init__ ###


    def shape(self):
        """ Returns the shape of grid_cells with the number of different elements

        Returns: the numpy shape (rows, cols) appended by the number of different 
            elements in the grid as a tuple, so: (# row, # cols, # different elements)
        """
        s = set(self.grid_cells.flatten())

        return (self.grid_cells[0], self.grid_cells[1], len(s))

    ### shape ###


    def save_grid(self, file_path: str):
        """ Saves the Grid to file.

        Args:
            file_path (str): name of file to save Grid to

        Returns:
            bool: True: save succeeded, else False
        """
        if not isinstance(file_path, str):
            logging.critical("Invalid file_path. It must be a string.")
            return False

        elif not os.path.exists(os.path.dirname(file_path)):
            logging.critical("Cannot find the directory for " + file_path)
            return False

        else:
            np.save(file_path, self.grid_cells, allow_pickle=False, fix_imports=True)
            return True

        # if

    ### save_grid ###


    @classmethod
    def load_grid(cls, file_path: str):
        """ Loads the Grid from file.

        Args:
            file_path (str): Name of file to read Grid from

        Returns:
            bool: True if load succeeded, else False
        """
        if not isinstance(file_path, str):
            logging.critical("Invalid file_path. It must be a string: " + file_path)
            return None

        elif not os.path.exists(file_path):
            logging.critical("Cannot find grid file: " + file_path)
            return None

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)

        # if

    ### load_grid ###

        
    def generate_grid(self, generator) -> None:
        """ Generates the Grid from a GridGenerator.

        Args:
            generator (GridGenerator): GridGenerator to generator Grid from
        """
        generator.generate(self)
         
        return

    ### generate_grid ###


    def print_grid(self, matrix) -> None:
        """ Creates text representation of the grid.

        Args:
            matrix (Object): _description_

        Returns:
            str: Grid in str format
        """
        cols, rows = matrix.shape
        strmat = 'Grid size (width x height): {:d} x {:d}\n'.format(cols, rows)
        
        for row in range(rows):
            line = ''
            for col in range(cols):
                line += '{:3d}'.format(matrix[col, row])
            # for
            strmat += line + '\n'
        # for
        
        return strmat

    ### print_grid ###

    
    def insert_thing(self, ThingClass, loc, erase: bool = False):
        """ Inserts new ThingClass in specific location. 

        Args:
            ThingClass (Thing): Type of thing to insert
            loc (tuple): (row, col) location
            erase (bool, optional): When True, delete all existing objects 
                loc before insertion. Defaults to False.

        Returns:
            _type_: _description_
        """
        # First: remove all objects from loc when erase is True.
        if erase:
            thing = self.find_thing_by_loc(loc)
            while thing is not None:
                self.remove_thing(thing)
                thing = self.find_thing_by_loc(loc)
            
            # while

        # if

        # x = ThingClass.__name__
        # create thing from provided ThingClass
        thing = ThingClass(loc, self.definitions, self)
        thing.Verbose = self.verbose

        # assign symbol to grid cells
        self.grid_cells[loc] = self.definitions.loc[thing.type, COL_CATEGORY]

        # insert thing into thing dictionary
        self.things_by_id[thing.id] = thing
        
        return thing
    
    ### insert_thing ###

    
    def insert_things(self, ThingClass, locs: list) -> list:
        """ Insert Thing at list of locations.

        Args:
            ThingClass (object): Class of thing to insert
            locs (list): List of locations

        Returns:
            list: _description_
        """
        things = []
        if not locs is None:            
            for loc in locs:
                thing = self.insert_thing(ThingClass, loc)
                things.append(thing)
        
        return things

    ### insert_things ###

    
    def add_thing(self, ThingClass, loc: tuple):
        """ Adds thing at end of turn.

        Args:
            ThingClass (Thing): Class of thing to insert.
            loc (tuple): (row, col) of location in Grid)
        """
        self.things_to_be_added_at_end_of_turn.append((ThingClass, loc))
    
    ### add_thing ###


    def set_tracker(self, thing) -> None:
        """ Sets Grid tracker to Thing to be tracked.

        Args:
            thing (Thing): Thing to be tracked
        """
        self.tracked = thing
        
        return

    ### set_tracker ###


    def set_start(self, ThingClass, loc: tuple) -> None:
        """ Set start location

        Args:
            ThingClass (Thing): Set start location of Grid.
            loc (tuple): (row, col)
        """
        # insert start location
        self.start = self.insert_thing(ThingClass, loc, erase = True)
        
        return

    ### set_start ###

    
    def set_destination(self, ThingClass, loc: tuple) -> None:
        # insert destination location
        self.destination = self.insert_thing(ThingClass, loc, erase = True)
        
        return

    ### set_destination ###


    def list_things(self):
        """ Show a list of all things of the grid.
        """
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            logger.info(f'{thing.id}: {thing.type}, {thing.category} {thing.location} '
                        f'mass={thing.mass:.2f} growth={thing.growth} age={thing.age}')

        return

    ### list_things ###

    
    def process_command(
            self, 
            command: str, 
            grid_pos: tuple, 
            definitions: pd.DataFrame
        ) -> None:
        """ Processes a command to operate on the Grid.

        Args:
            command (str): One-character command
            grid_pos (tuple): Position for the command to operate on
            definitions (pd.DataFrame): List of Thing definitions
        """
    
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

    
    def find_thing_by_loc(self, loc: tuple, type = None):
        """ Find Things at specific location.

        Args:
            loc (tuple): Location to find Things
            type (Thing, optional): Specific type of thing to look for. 
                Defaults to None.

        Returns:
            Thing: first thing found will be returned
        """
        for key in self.things_by_id.keys():
            thing = self.things_by_id[key]
            if loc == thing.location:
                if type is None:
                    return thing

                else:
                    if thing.type == type:
                        return thing

                    # if
                # if
            # if
        # for

        return None
    
    ### find_thing_by_loc ###

    
    def find_thing_by_type(self, thing_type: str):
        """ Find specific type of thing.

        Args:
            thing_type (str): Type of Thing

        Returns:
            Thing: First Thing found or None
        """
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
        """ Remove and delete specific thing.

        Args:
            thing (Thing): Thing to be removed
        """
        if thing is None:
            logger.warning('Grid.remove_thing: argument is None')

        else:
            id = thing.id
            if id in self.things_by_id.keys():
                if thing.category == self.grid_cells[thing.location]:
                    self.grid_cells[self.things_by_id[id].location] = \
                        self.definitions.loc['Field', COL_CATEGORY]
                        
                del self.things_by_id[id]

                if self.verbose > 0:
                    logger.info(str(thing.type) + ' removed: ' + str(id))
            else:
                logger.warning('No ' + str(thing.type) + ' found: ' + str(id))
            
        return

    ### remove_thing ###

    
    def move_things(self):
        """ Move all things based on their internal move function.

        As a result of move, things can be deleted. They are flagged as 
        to be deleted and at the end of all moves the Thing to remove
        will be deleted.
        """
        self.turns += 1

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
        """ Move all things and at the end insert or remove things when needed.
        """
        self.turns += 1

        # Move all things
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            thing.next_turn()

        # add things to grid because of end of turn
        if len(self.things_to_be_added_at_end_of_turn) > 0:
            for addition in self.things_to_be_added_at_end_of_turn:
                self.insert_thing(addition[0], addition[1])

            self.things_to_be_added_at_end_of_turn = []
            
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
        """ Returns True when destination has been reached.

        Args:
            max_turns (int, optional): Maximum number of turns after which
                always True is returned. When zero max_turns will never be reached. 
                Defaults to 0.

        Returns:
            bool: True when destionation reached or turns >- max_turns
        """

        if max_turns > 0 and self.turns >= max_turns:
            return True

        elif self.tracked.location[0] == self.destination.location[0] and \
           self.tracked.location[1] == self.destination.location[1]:

            return True

        else:
            return False

    ### destination_reached ###

    
    def get_n_things(self, type_str) -> int:
        """ Returns number of Things of specific type.

        Args:
            type_str (str): Type of Things to be counted

        Returns:
            int: Number of things of specific type
        """
        n = 0
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            if thing.type == type_str:
                n += 1
                
        return n

    ### get_n_things ###
    
       
    def get_vehicles_mass(self, ThingClass):
        mass = 0
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            if isinstance(thing, ThingClass):
                mass += thing.mass
                
        return mass

    ### get_vehicles_mass ###

    def load_thing_definitions(self, res_path: str, style: int):
        # load the csv file with Thing definitions into dataframe
        filename = '../config/things.csv' # os.path.join(res_path, 'config/things.csv')
        definitions = pd.read_csv(filename, sep=';', index_col='Name')

        # next load icon info from images is resource path
        for key in definitions.index:
            filename = key.lower() + '-' + str(style) + '.png'
            filename = os.path.join('images', filename)
            filename = os.path.join(res_path, filename)
            
            definitions.loc[key, 'Icon'] = pygame.image.load(filename)

        # for

        logger.info(str(definitions))

        return definitions

    ### load_thing_definitions ###

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

        if self.verbose > 0:
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
    
