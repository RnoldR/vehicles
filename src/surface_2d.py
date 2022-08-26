import os
import time
import yaml
import pygame
import random
import numpy as np

# Code initialisatie: logging
import logging
import importlib
importlib.reload(logging)

# create logger
logger = logging.getLogger('distances')

logger.setLevel(10)

# create file handler which logs even debug messages
fh = logging.FileHandler('grid-view-2D.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(logging.Formatter('%(message)s'))

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# Initialize Pandas  display options such that the whole DataFrame is printed
#pd.options.display.max_rows = 999999
#pd.options.display.max_columns = 999999

# Vehicle directions
COMPASS = {"N": (0, -1),
           "E": (1, 0),
           "S": (0, 1),
           "W": (-1, 0),
           "X": (0, 0)
          }

# choose icon style (1-3)    
ICON_STYLE = 1

def load_config(filename):
    """
    Load configuration from taml file

    filename (str): name of configuration file
    """

    filename = os.path.join(res_path, 'config/config.yaml')
    with open(filename) as yaml_data:
        config = yaml.load(yaml_data, Loader=yaml.FullLoader)
    
    for key, value in config['Things'].items():
        logger.debug(str(key) + ': ' + str(value) + ', ID =' +
                     str(value[0]) + ', cost = ' + str(value[1]))
        
    return config['Things']

class GridView2D:
    def __init__(self, grid_name: str="Grid2D", grid_file_path: str=None, 
                 res_path: str=None, grid_size=(30, 30), screen_size=(600, 600),
                 has_loops: bool=False, init_pos=(1, 1),
                 n_mushrooms: int=0, n_cactuses: int=0, n_rocks: int=0):
        
        caption: str = '{:s} - {:d} x {:d}'.format(grid_name, grid_size[0], grid_size[1])

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.__game_over: bool = False
        self.turns: int = 0
        self.direction = None
        self.insert: str = ' '

        # Load a grid
        if grid_file_path is None:
            self.__grid = Grid(grid_size=grid_size, has_loops=has_loops, 
                               n_mushrooms=n_mushrooms, 
                               n_cactuses=n_cactuses,
                               n_rocks=n_rocks)
        else:
            if not os.path.exists(grid_file_path):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "grid_samples", grid_file_path)
                
                if os.path.exists(rel_path):
                    grid_file_path = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % grid_file_path)
                    
            self.__grid = Grid(grid_cells=Grid.load_grid(grid_file_path))

        self.grid_size = self.__grid.grid_size
        
        self.compute_screen_size(screen_size, grid_size)
        
        # to show the right and bottom border
        self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))

        # load icons
        self.load_resources(res_path, ICON_STYLE)
        
        # Create the Robot
        self.init_pos = init_pos
        self.__robot = self.grid.insert_thing(Vehicle, self.init_pos)

        # Create a background
        #self.background = pygame.Surface((self.SIM_W, self.SIM_H)).convert()
        #self.background.fill((255, 255, 255))

        # Create a layer for the grid
        self.grid_layer = pygame.Surface((self.SIM_W, self.SIM_H)).convert()#.convert_alpha()
        self.grid_layer.fill((255, 255, 255, 0,))
        
        self.interface = pygame.Surface((self.INTF_W, self.INTF_H)).convert()
        self.interface.fill((0, 0, 255, 0,))
        
        # Create sidebar
        self.interface = pygame.Surface((self.INTF_W, self.INTF_H))

        # show the grid
        self.background = self.create_background()

        # show all things
        self.__draw_things()

        return
    
    def compute_screen_size(self, screen_size, grid_size):
        grid_w: int = int(screen_size[0] / grid_size[0] + 0.5)
        grid_h: int = int(screen_size[1] / grid_size[1] + 0.5)
        self.CELL_W: int = min(grid_w, grid_h)
        self.CELL_H: int = self.CELL_W
        self.SIM_W: int = grid_size[0] * self.CELL_W - 1
        self.SIM_H: int = grid_size[1] * self.CELL_H - 1
        self.INTF_W: int = 300
        self.INTF_H: int = self.SIM_H
        self.SCREEN_W: int = self.SIM_W + self.INTF_W
        self.SCREEN_H: int = self.SIM_H

        return 
    
    def load_resources(self, res_path, style):
        for key in INFO:
            filename = key.lower() + '-' + str(style) + '.png'
            filename = os.path.join(res_path, filename)
            
            img = pygame.image.load(filename)
            
            img = pygame.transform.scale(img, (self.CELL_W, self.CELL_H)).convert_alpha()
            INFO[key].append(img)
            
        return
    
    def get_events(self):
        self.direction = "X" # Equals to don't move
        waiting = True
        while waiting:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                logger.debug(str(event))
                if event.key == pygame.K_ESCAPE:
                    self.__game_over = True
                    waiting = False
                elif event.key == pygame.K_LEFT:
                    self.direction = "W"
                    waiting = False
                elif event.key == pygame.K_RIGHT:
                    self.direction = "E"
                    waiting = False
                elif event.key == pygame.K_UP:
                    self.direction = "N"
                    waiting = False
                elif event.key == pygame.K_DOWN:
                    self.direction = "S"
                    waiting = False
                elif event.key == pygame.K_m:
                    self.insert = 'm'
                elif event.key == pygame.K_c:
                    self.insert = 'c'
                elif event.key == pygame.K_r:
                    self.insert = 'r'
                elif event.key == pygame.K_v:
                    self.insert = 'v'
                elif event.key == pygame.K_w:
                    self.insert = 'w'
                elif event.key == pygame.K_f:
                    self.insert = 'f'
    
            elif event.type == pygame.QUIT:
                self.__game_over = True
                waiting = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                grid_pos = self.__pixel_to_pos(event.pos)
                logger.info('converting ' + str(event.pos) + ' to ' + str(grid_pos))
                if self.insert in ['m', 'c', 'r', 'v', 'w']:
                    if self.grid.grid_cells[grid_pos] != 0:
                        logger.warning('Already occupied ' + str(grid_pos))
                    else:
                        if self.insert == 'm':
                            thing = self.grid.insert_thing(Mushroom, grid_pos)
                        elif self.insert == 'c':
                            thing = self.grid.insert_thing(Cactus, grid_pos)
                        elif self.insert == 'r':
                            thing = self.grid.insert_thing(Rock, grid_pos)
                        elif self.insert == 'v':
                            thing = self.grid.insert_thing(Vehicle, grid_pos)
                        elif self.insert == 'w':
                            thing = self.grid.insert_thing(Wall, grid_pos)

                        logger.info('Inserted ' + str(thing.type) + ' at ' +
                                    str(thing.location))
                        
                elif self.insert == 'f':
                    thing = self.grid.find_thing_by_loc(grid_pos)
                    self.grid.remove_thing(thing)
                    if thing is None:
                        logger.info('Nothing found')
                    else:
                        logger.info('Removed ' + str(thing.type) + ' at ' +
                                    str(thing.location))
                
                self.__view_update()
                        
        return

    def move_things(self):
        if self.direction is None:
            self.grid.move_things()
        else:
            self.grid.move_things(self.direction)
        
        return
    
    def show_status(self, mess: str):
        # initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
        myfont = pygame.font.SysFont("sans", 20)

        # render text
        label = myfont.render(mess, 1, (0, 0, 0))
        self.interface.fill((255, 255, 255))
        self.interface.blit(label, (5, 5))
    
    def update_screen(self, mode="human"):
        """ Updates all changes to the grid
        
        Args:
            mode (str): "human" shows all moves directly on the screen
            
        Returns:
            None
        """
        self.turns += 1
                
        self.show_status('Turn: ' + str(self.turns))
        
        caption = 'Turn: {:d} Energy {:.2f} - {:s}'.format(self.turns,
                         self.grid.get_vehicles_energy(), str(self.__robot.location))
        pygame.display.set_caption(caption)
        logger.debug ('')
        logger.debug('*** ' + caption)

        try:
            if not self.__game_over:
                # set the background
                self.grid_layer.blit(self.background, (0, 0))

                # Draw all things
                self.__draw_things()
                
                # update the screen
                self.screen.blit(self.grid_layer, (0, 0))
                self.screen.blit(self.interface, (self.SIM_W + 1, 0))

                if mode == "human":
                    pygame.display.flip()

                if self.grid.get_vehicles_energy() <= 0:
                    self.quit_game()

        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e

        return 

    def quit_game(self):
        """ 
        Quits the game
        """
        try:
            self.__game_over = True
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
        
        return

    def reset_robot(self):
        """ 
        Resets the robot
        """
        self.__robot.location = self.init_pos
        
        return
   
    '''
    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()
                    
        return
            #self.__controller_update()
    '''
    
    def __draw_wall(self, layer, cell):
        layer.blit(INFO["Wall"][2], (self.CELL_W * cell[0], self.CELL_H * cell[1]))

    def create_background(self):
        line_color = (0, 0, 0, 255)
        background = pygame.Surface((self.SIM_W, self.SIM_H)).convert()
        background.fill((255, 255, 255))

        # drawing the horizontal lines
        for y in range(self.grid.GRID_H + 2):
            pygame.draw.line(background, line_color, (0, y * self.CELL_H),
                             (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.grid.GRID_W + 2):
            pygame.draw.line(background, line_color, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H + self.CELL_H))
            
        # creating the walls
        for x in range(len(self.grid.grid_cells)):
            for y in range (len(self.grid.grid_cells[x])):
                cell = (x, y)
                status = self.grid.grid_cells[x, y]
                if status == INFO["Wall"][0]:
                    self.__draw_wall(background, cell)

        return background
    
    def __pixel_to_pos(self, pixel):
        x = int(pixel[0] / self.CELL_W)
        y = int(pixel[1] / self.CELL_H)
        
        logger.debug('w, h ' + str(self.grid.GRID_W) + ', ' + str(self.grid.GRID_H))
        
        return (x, y)

    def __draw_things(self, transparency=160):
        for key in self.grid.things_by_id.keys():
            thing = self.grid.things_by_id[key]
            self.__draw_bitmap(thing.type, thing.location)
            
        return
    
    def __draw_bitmap(self, cat, cell):
        self.grid_layer.blit(INFO[cat][2], (self.CELL_W * cell[0], self.CELL_H * cell[1]))
    
    @property
    def robot(self):
        return self.grid.robot

    @property
    def grid(self):
        return self.__grid

    @property
    def game_over(self):
        return self.__game_over

## Class: GridView2D ##

class Grid:
    def __init__(self, grid_cells=None, grid_size=(10,10), has_loops=True, 
                 n_vehicles=1, n_rocks = 0,
                 n_mushrooms=0, p_mushrooms=0.0,
                 n_cactuses=0, p_cactuses=0.0):
        
        # grid member variables
        self.grid_cells = grid_cells
        self.has_loops = has_loops
        self.things_by_id = {}
        self.vehicles_by_id = {}
        self.n_vehicles = n_vehicles
        self.n_rocks = n_rocks
        self.n_mushrooms = n_mushrooms
        self.p_mushrooms = p_mushrooms
        self.n_cactuses = n_cactuses
        self.p_cactuses = p_cactuses

        # Use existing grid if exists
        if self.grid_cells is not None:
            if isinstance(self.grid_cells, (np.ndarray, np.generic)) and len(self.grid_cells.shape) == 2:
                self.grid_size = tuple(grid_cells.shape)
            else:
                raise ValueError("grid_cells must be a 2D NumPy array.")
                
        # Otherwise, generate a random grid
        else:
            # grid's configuration parameters
            if not (isinstance(grid_size, (list, tuple)) and len(grid_size) == 2):
                raise ValueError("grid_size must be a tuple: (width, height).")
            self.grid_size = grid_size

            self.generate_grid()
        # if
            
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
        
    def generate_grid(self):
        # list of all cell locations
        self.grid_cells = np.zeros(self.grid_size, dtype=int)
        
        # Create a wall around the grid
        for x in range(self.grid_size[0]):
            self.insert_thing(Wall, (x, 0))
            self.insert_thing(Wall, (x, self.GRID_H-1))
            
        for y in range(self.grid_size[1]):
            self.insert_thing(Wall, (0, y))
            self.insert_thing(Wall, (self.GRID_W-1, y))
        
        self.insert_things(Mushroom, self.generate_random_locs(self.n_mushrooms))
        self.insert_things(Cactus, self.generate_random_locs(self.n_cactuses))
        self.insert_things(Rock, self.generate_random_locs(self.n_rocks))
        self.insert_thing(Start, (1, 1))
        self.START = self.insert_thing(Destination, (self.GRID_W-2, self.GRID_H-2))
        self.DESTINATION = self.print_grid()
         
        return

    def print_grid(self):
        cols, rows = self.grid_size
        print('Grid size (width x height): ', cols, 'x', rows)
        for row in range(rows):
            for col in range(cols):
                print(self.grid_cells[col, row], end=' ')
                
            print()
                
        return
    
    def insert_thing(self, ThingClass, loc):
        thing = ThingClass(loc)
        self.grid_cells[loc] = INFO[thing.type][0]
        self.things_by_id[thing.id] = thing
        
        return thing
    
    def insert_things(self, ThingClass, locs):
        things = []
        if not locs is None:            
            for loc in locs:
                thing = self.insert_thing(ThingClass, loc)
                things.append(thing)
        
        return things
    
    def generate_random_locs(self, num_things):
        # Check if things should be generated
        if num_things <= 0:
            return
        
        # Generate all available cells
        cell_ids = [(x, y) for y in range(self.GRID_H) 
                            for x in range(self.GRID_W) 
                                if self.grid_cells[x, y] == 0]
    
        # limit the maximum number of things to half the number of cells available.
        max_things = int(self.GRID_W * self.GRID_H / 2)
        num_things = min(max_things, num_things)
        thing_locations = random.sample(cell_ids, num_things)

        return thing_locations
    
    def find_thing_by_loc(self, loc):
        for key in self.things_by_id.keys():
            if loc == self.things_by_id[key].location:
                return self.things_by_id[key]
        
        return None
    
    def remove_thing(self, thing):
        if thing is None:
            logger.warning('Grid.remove_thing: argument is None')
        else:
            id = thing.id
            if id in self.things_by_id.keys():
                self.grid_cells[self.things_by_id[id].location] = INFO['Field'][0]
                del self.things_by_id[id]
                logger.info(str(thing.type) + ' removed: ' + str(id))
            else:
                logger.warning('No ' + str(thing.type) + ' found: ' + str(id))
            
        return
    
    def move_things(self, direction=None):
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            if isinstance(thing, Vehicle) and not direction is None:
                thing.move(self, direction)
            else:
                thing.move(self)
            
        removes = []
        for id in self.things_by_id:
            if self.things_by_id[id].deleted:
                removes.append(id)
            
        for id in removes:
            thing = self.things_by_id[id]
            self.remove_thing(thing)

        return
    
    def get_n_things(self, type_str):
        n = 0
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            if thing.type == type_str:
                n += 1
                
        return n
       
    def get_vehicles_energy(self):
        energy = 0
        for id in self.things_by_id:
            thing = self.things_by_id[id]
            if isinstance(thing, Vehicle):
                energy += thing.energy
                
        return energy
    
    @property
    def robot(self):
        for id in self.things_by_id:
            if isinstance(self.things_by_id[id], Vehicle):
                return self.things_by_id[id]
            
        return None
    
    @property
    def GRID_W(self):
        return int(self.grid_size[0])

    @property
    def GRID_H(self):
        return int(self.grid_size[1])

## Class: Grid ##
        
class Thing():
    Seq = 0
    
    def __init__(self, location=(1, 1)):
        Thing.Seq += 1

        self.id = Thing.Seq
        self.location = location
        self.type = 'Field'
        self.category = INFO[self.type][0]
        self.energy = 0
        self.deleted = False
        
        return
    
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
        cost = 0
        may_move = False
        
        if idx == INFO['Field'][0]:
            cost = INFO['Field'][1]
            may_move = True
        elif idx == INFO['Wall'][0]:
            cost = INFO['Wall'][1]
            may_move = False
        elif idx == INFO['Vehicle'][0]:
            thing = grid.find_thing_by_loc(potential_pos)
            cost = thing.energy
            may_move = False
        elif idx == INFO['Mushroom'][0]:
            thing = grid.find_thing_by_loc(potential_pos)
            cost = thing.energy
            may_move = False
        elif idx == INFO['Cactus'][0]:
            thing = grid.find_thing_by_loc(potential_pos)
            cost = thing.energy
            may_move = False
        elif idx == INFO['Rock'][0]:
            thing = grid.find_thing_by_loc(potential_pos)
            cost, may_move = thing.cost(grid, direction)
            cost = -abs(cost) + thing.energy
        elif idx == INFO['Start'][0]:
            cost = INFO['Start'][1]
            may_move = True
        elif idx == INFO['Destination'][0]:
            cost = INFO['Destination'][1]
            may_move = True
        else:
            raise ValueError('*** Unknown field code in Rock.move:', idx)
            
        return cost, may_move
    
    def move(self, grid, direction=None):
        return
    
## Class: Thing ##

## Class: Wall ##

class Wall(Thing):
    def __init__(self, location):
        super().__init__(location)

        self.type = 'Wall'
        self.category = INFO[self.type][0]
        self.energy = INFO[self.type][1]
        
        return

class Vehicle(Thing):
    def __init__(self, location=(1, 1)):
        super().__init__(location)
        
        self.type = 'Vehicle'
        self.category = INFO[self.type][0]
        self.energy = INFO[self.type][1]
        
        return
    
    def move(self, grid, direction=None):
        if direction is None:
            direction = random.sample(['N', 'E', 'S', 'W'], 1)[0]
            
        potential_loc = (self.location[0] + COMPASS[direction][0], self.location[1] + COMPASS[direction][1])
        idx = grid.grid_cells[potential_loc]
        cost, may_move = self.cost(grid, direction)
        
        # Vehicle may have reached destination
        if idx == INFO['Destination'][0]:
            new_loc = potential_loc
            logger.info('!!!Destination reached!!!')
                    
        # Vehicle may move over the field
        elif idx == INFO['Field'][0]:
            new_loc = potential_loc
            
        # Vehicle may not move thru a wall
        elif idx == INFO['Wall'][0]:
            new_loc = self.location
            logger.info('Vehicle cost from Wall: ' + str(cost))
            
        # Rock cannot be pushed thru a Vehicle
        elif idx == INFO['Vehicle'][0]:
            thing = grid.find_thing_by_loc(potential_loc)
            new_loc = self.location
            
        # Cannot move over a mushroom which is lost
        elif idx == INFO['Mushroom'][0]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.deleted = True
            new_loc = self.location
            logger.info('Vehicle energy from Mushroom: ' + str(cost))
            
        # Cannot be moved over a cactus which remainslost
        elif idx == INFO['Cactus'][0]:
            thing = grid.find_thing_by_loc(potential_loc)
            new_loc = self.location
            logger.info('Vehicle cost from Cactus: ' + str(cost))
            
        # Rock can move, depending on the object before it
        elif idx == INFO['Rock'][0]:
            new_loc = self.location
            if may_move:
                thing = grid.find_thing_by_loc(potential_loc)
                thing.move(grid, direction)
                new_loc = potential_loc

            logger.info('Vehicle cost from Rock: ' + str(cost))

        else:
            logger.critical('*** Unknown field code in Rock.move: ' + str(idx))
            raise ValueError('*** Unknown field code in Rock.move:', idx)
                
        # if
    
        self.energy += cost
        grid.grid_cells[self.location] = INFO['Field'][0]
        self.location = new_loc
        grid.grid_cells[self.location] = INFO['Vehicle'][0]
        
        return cost, self.location
            
## Class: Vehicle ##

class Mushroom(Thing):
    def __init__(self, location):
        super().__init__(location)

        self.type = 'Mushroom'
        self.category = INFO[self.type][0]
        self.energy = INFO[self.type][1]
        
        return

## Class: Mushroom ##

class Cactus(Thing):
    def __init__(self, location):
        super().__init__(location)

        self.type = 'Cactus'
        self.category = INFO[self.type][0]
        self.energy = INFO[self.type][1]
        
        return

## Class: Mushroom ##

class Rock(Thing):
    def __init__(self, location):
        super().__init__(location)

        self.type = 'Rock'
        self.category = INFO[self.type][0]
        self.energy = INFO[self.type][1]
        
        return

    def move(self, grid, direction=None):
        # When direction is None this function is called to move itself, 
        # not from vehicle move (pushing the rock). In that case it returns
        # immediately as it does not spontaneously move
        if direction is None:
            return
            
        # Compute a move based on the push of a vehicle
        potential_loc = (self.location[0] + COMPASS[direction][0], self.location[1] + COMPASS[direction][1])
        idx = grid.grid_cells[potential_loc]
        cost, new_loc = self.cost(grid, direction)
        thing = None
        
        # Rock may move of the field
        if idx == INFO['Field'][0]:
            new_loc = potential_loc
            
        # Rock may not move thru a wall
        elif idx == INFO['Wall'][0]:
            new_loc = self.location
            
        # Rock cannot be pushed thru a wall
        elif idx == INFO['Vehicle'][0]:
            thing = grid.find_thing_by_loc(potential_loc)
            new_loc = self.location
            
        # Can be pushed over a mushroom which is lost
        elif idx == INFO['Mushroom'][0]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.deleted = True
            new_loc = potential_loc
            
        # Can be pushed over a cactus which is lost
        elif idx == INFO['Cactus'][0]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.deleted = True
            new_loc = potential_loc
            
        # Rock can move, depending on the object before it
        elif idx == INFO['Rock'][0]:
            thing = grid.find_thing_by_loc(potential_loc)
            thing.move(grid, direction)
            new_loc = potential_loc
        else:
            raise ValueError('*** Unknown field code in Rock.move:', idx)
            
        # if
        if not thing is None:
            logger.info('Rock added cost from ' + str(thing.type) + 
                        ' cost = ' + str(cost))
    
        grid.grid_cells[self.location] = INFO['Field'][0]
        self.location = new_loc
        grid.grid_cells[self.location] = INFO['Rock'][0]
            
        return cost, self.location
            
## Class: Rock ##
        
class Start(Thing):
    def __init__(self, location):
        super().__init__(location)

        self.type = 'Start'
        self.category = INFO[self.type][0]
        self.energy = INFO[self.type][1]
        
        return
    
## Class: Start ##

class Destination(Thing):
    def __init__(self, location):
        super().__init__(location)

        self.type = 'Destination'
        self.category = INFO[self.type][0]
        self.energy = INFO[self.type][1]
        
        return
    
## Class: Start ##

def test_move_around():
    grid = GridView2D(screen_size=(500, 500), grid_size=(20, 15), 
                      res_path=os.path.join(res_path, 'images'), 
                      n_mushrooms=5, n_cactuses=4, n_rocks=3)
    
    cols, rows = grid.grid.grid_size
    print(grid.grid.print_grid())

    grid.update_screen()
    grid.direction = "X"
    time.sleep(1)
    try:
        while not grid.game_over:
            grid.get_events()
            grid.move_things()
            grid.update_screen()
            #time.sleep(1)
   
    finally:
        time.sleep(2)
        pygame.quit()
        
    return
    
def test_move_rock(p):
    init_pos = p[0]
    grid = GridView2D(screen_size=(500, 500), grid_size=(20, 15), 
                      res_path=res_path, 
                      n_mushrooms=0, n_cactuses=0, n_rocks=0,
                      init_pos = init_pos)
    
    cols, rows = grid.grid.grid_size

    direction = "W"
    thing1 = grid.grid.insert_thing(Rock, p[1])
    thing2 = grid.grid.insert_thing(Rock, p[2])
    thing3 = grid.grid.insert_thing(Rock, p[3])

    print(grid.grid.print_grid())
    grid.update_screen()
    time.sleep(1)

    robot = grid.robot
    start_energy = robot.energy
    grid.robot.move(grid.grid, direction)
    grid.update_screen()

    exp_loc = init_pos#(self.init_pos[0] + m2d.COMPASS[direction][0], self.init_pos[1] + m2d.COMPASS[direction][1])
    loc = robot.location
    thing_energy = thing1.energy + thing2.energy + thing3.energy
    exp_energy = start_energy + thing_energy
    energy = robot.energy
    
    if loc != exp_loc:
        logger.warn('Incorrect location, expected ' + str(exp_loc) + 
                    ' not ' + str(loc))
    if energy != exp_energy:
        logger.warning('Incorrect energy, expected '+ str(exp_energy) +
                       ' not ' + str(energy))
    
    time.sleep(3)
    pygame.quit()
        
    return

def test_random():
    grid = GridView2D(screen_size=(500, 500), grid_size=(20, 15), 
                      res_path=res_path, 
                      n_mushrooms=5, n_cactuses=4, n_rocks=3)
    
    cols, rows = grid.grid.grid_size
    print(grid.grid.print_grid())

    grid.update_screen()
    time.sleep(1)
    try:
        while not grid.game_over:
            grid.move_things()
            grid.update_screen()
            time.sleep(0.2)
   
    finally:
        time.sleep(2)
        pygame.quit()
        
    return
    
if __name__ == "__main__":
    res_path='/media/i/home/arnold/development/python/machine_learning/grid2d'
    #res_path = os.path.join(res_path, 'images')

    INFO = load_config(res_path)

    test_move_around()
