# Code initialisatie: logging
import logging
logger = logging.getLogger()

import os
import time
from pygame.constants import K_p
import yaml
import pygame
import random
import numpy as np
import pandas as pd

from grid import Grid

from grid_thing_data import COL_ICON

# Initialize Pandas  display options such that the whole DataFrame is printed
pd.options.display.max_rows = 999999
pd.options.display.max_columns = 999999

class GridView2D:
    def __init__(self, grid: Grid, definitions: pd.DataFrame, 
                 title: str='', screen_size=(600, 600)):
        # Initialize PyGame 
        pygame.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        pygame.key.set_repeat(500, 100)
        self.robot = None

        # set grid        
        self.grid = grid
        self.grid_size = self.grid.grid_size
        self.definitions = definitions

        # set viewer variables
        self.game_over: bool = False
        self.insert: str = ' '
        
        # set the size of all elements of the viewer
        self.compute_screen_size(screen_size, self.grid_size)
        
        # to show the right and bottom border
        self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))

        # transform the images to the new cell size
        for idx in self.definitions.index:
            img = self.definitions.loc[idx, COL_ICON].copy()
            self.definitions.loc[idx, COL_ICON] = pygame \
                .transform.scale(img, (self.CELL_W, self.CELL_H)).convert_alpha()
        
        #create a Surface for the grid
        self.grid_layer = pygame.Surface((self.SIM_W, self.SIM_H)).convert()#.convert_alpha()
        self.grid_layer.fill((255, 255, 255, 0,))

        # create interface Surface        
        self.interface = pygame.Surface((self.INTF_W, self.INTF_H)).convert()
        self.interface.fill((0, 0, 255, 0,))
        self.interface = pygame.Surface((self.INTF_W, self.INTF_H))
        
        # create a background
        self.background = self.create_background()

        # show all things
        self.__draw_things()

        # set a caption
        caption: str = '{:s} - {:d} x {:d}' \
                        .format(title, self.grid_size[0], self.grid_size[1])
        pygame.display.set_caption(caption)

        return
    
    # __init__ #
    
    def compute_screen_size(self, screen_size: int, grid_size: int):
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
    
    # compute_screen_size #
    
    def get_events(self):
        #self.direction = "X" # Equals to don't move
        waiting = True
        k = pygame.K_p
        while waiting:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                logger.debug(str(event))
                if event.key == pygame.K_ESCAPE:
                    self.game_over = True
                    waiting = False
                elif event.key == pygame.K_LEFT:
                    self.grid.tracked.direction = "W"
                    waiting = False
                elif event.key == pygame.K_RIGHT:
                    self.grid.tracked.direction = "E"
                    waiting = False
                elif event.key == pygame.K_UP:
                    self.grid.tracked.direction = "N"
                    waiting = False
                elif event.key == pygame.K_DOWN:
                    self.grid.tracked.direction = "S"
                    waiting = False
                elif event.key == pygame.K_p:
                    self.grid.process_command('P', None, self.definitions)
                elif event.key == pygame.K_MINUS:
                    self.grid.process_command('-', None, self.definitions)
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
                # if
            
            elif event.type == pygame.QUIT:
                self.game_over = True
                waiting = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                grid_pos = self.__pixel_to_pos(event.pos)
                logger.info('converting ' + str(event.pos) + ' to ' + str(grid_pos))
                self.grid.process_command(self.insert, grid_pos, self.definitions)
                
                self.__view_update()
                
            # if
                        
        return

    ### get_events ###

    def move_things(self):
        self.grid.move_things()
        
        return

    ### move_things ###
    
    def next_turn(self):
        self.grid.next_turn()
        
        return

    ### next_turn ###
    
    def blit_text(self, surface, text, pos, font, color=pygame.Color('black')):
        # 2D array where each row is a list of words.
        words = [word.split(' ') for word in text.splitlines()]  

        # The width of a space.
        space = font.size(' ')[0]

        max_width, max_height = surface.get_size()
        x, y = pos
        for line in words:
            for word in line:
                word_surface = font.render(word, 0, color)
                word_width, word_height = word_surface.get_size()

                if x + word_width >= max_width:
                    x = pos[0]  # Reset the x.
                    y += word_height  # Start on new row.

                # if

                surface.blit(word_surface, (x, y))
                x += word_width + space

            # for

            x = pos[0]  # Reset the x.
            y += word_height  # Start on new row.

        # for

        return

    ### blit_text ###

    def show_status(self, mess: str):
        # initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
        myfont = pygame.font.SysFont("sans", 18)

        # fill surface with white
        self.interface.fill((255, 255, 255))

        # show the text
        self.blit_text(self.interface, mess, (5, 5), myfont)

    ### show_status ###
    
    def update_screen(self, mode="human"):
        """ Updates all changes to the grid
        
        Args:
            mode (str): "human" shows all moves directly on the screen
            
        Returns:
            None
        """
        
        vehicle = self.grid.tracked
        if vehicle is not None:
            text = f'Turn: {self.grid.turns}\n----------\n\n'
            text += f'Vehicle {vehicle.id}\n\n'
            text += f'  Location ({vehicle.location[0]}, '
            text += f'{vehicle.location[1]})\n'
            text += f'  Mass: {vehicle.mass:.2f}\n'

            self.show_status(text)
            #self.txt(f'Mass: {self.grid.tracked.mass}', (5, 100))
            caption = 'Turn: {:d} Mass {:.2f} - {:s}'.format(self.grid.turns,
                            vehicle.mass, str(vehicle.location))
                         
            pygame.display.set_caption(caption)
            logger.debug ('')
            logger.debug('*** ' + caption)

        # if

        try:
            if not self.game_over:
                # set the background
                self.grid_layer.blit(self.background, (0, 0))

                # Draw all things
                self.__draw_things()
                
                # update the screen
                self.screen.blit(self.grid_layer, (0, 0))
                self.screen.blit(self.interface, (self.SIM_W + 1, 0))

                if mode == "human":
                    pygame.display.flip()

        except Exception as e:
            self.game_over = True
            self.quit_game()
            raise e

        # try..except

        return 

    ### update_screen ###

    def quit_game(self):
        """ 
        Quits the game
        """
        try:
            self.game_over = True
            pygame.display.quit()
            pygame.quit()

        except Exception:
            pass
        
        return

    ### quit_game ###

    def reset_robot(self):
        """ 
        Resets the robot
        """
        self.__robot.location = self.init_pos
        
        return
   
    def print_screen():
        
        return
        
    def __draw_wall(self, layer, cell):
        layer.blit(self.definitions["Wall"][2], (self.CELL_W * cell[0], self.CELL_H * cell[1]))

    def create_background(self):
        line_color = (0, 0, 0, 255)
        background = pygame.Surface((self.SIM_W, self.SIM_H)).convert()
        background.fill((255, 255, 255))

        # drawing the horizontal lines
        for y in range(self.grid.grid_size[1] + 2):
            pygame.draw.line(background, line_color, (0, y * self.CELL_H),
                             (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.grid.grid_size[0] + 2):
            pygame.draw.line(background, line_color, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H + self.CELL_H))
        '''
        # creating the walls
        for x in range(len(self.grid.grid_cells)):
            for y in range (len(self.grid.grid_cells[x])):
                cell = (x, y)
                status = self.grid.grid_cells[x, y]
                if status == self.definitions["Wall"][0]:
                    self.__draw_wall(background, cell)
        '''
        return background
    
    def __pixel_to_pos(self, pixel):
        x = int(pixel[0] / self.CELL_W)
        y = int(pixel[1] / self.CELL_H)
        
        logger.debug('w, h ' + str(self.grid.grid_size[0]) + ', ' + str(self.grid.grid_size[1]))
        
        return (x, y)

    def __draw_things(self, transparency=200):
        for key in self.grid.things_by_id.keys():
            thing = self.grid.things_by_id[key]
            self.__draw_bitmap(thing.type, thing.location)
            
        return
    
    def __draw_bitmap(self, cat, cell):
        self.grid_layer.blit(self.definitions.loc[cat, COL_ICON], 
                             (self.CELL_W * cell[0], self.CELL_H * cell[1]))
    
## Class: GridView2D ##
