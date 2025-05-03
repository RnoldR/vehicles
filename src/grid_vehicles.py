# Code initialisatie: logging
import logging
logger = logging.getLogger()

import os
import sys
import time
import random
import numpy as np
import pandas as pd

from math import sqrt
from statistics import mean

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential

from grid import Grid
from grid_thing import Thing
from grid_thing_data import COMPASS
from grid_objects import Vehicle, Start, Destination
from grid_sensors import Eye
from grid_thing_data import COL_CATEGORY, ACTIONS, COMPASS


class NeuralQ(Vehicle):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)
        
        # self.type = 'Vehicle'
        # self.category = self.definitions.loc[self.type, COL_CATEGORY]
        # self.mass = self.definitions.loc[self.type, COL_MASS]
        self.direction = 'X'

        # create basic sensors
        self.sensors = [
                        Eye(self, grid, definitions.loc['Wall', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['Mushroom', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['Cactus', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['Destination', COL_CATEGORY]),
                        Eye(self, grid, definitions.loc['NeuralQ', COL_CATEGORY]),
                       ]

        input_shape = self.grid_cells[0], self.grid_cells[1], len(self.sensors) # (6, 7, 3) # 3 values: 1) the actual piece, 2) the current player, 3) The player who is playing
        number_of_actions = len(COMPASS) # 7

        model = Sequential([
            Conv2D(16, (4,4), activation="relu", input_shape=input_shape),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(32, activation="relu"),
            Dense(number_of_actions)
        ])
        
        return
    
    ### __init__ ###

    def set_weights(self,w_wall: float, w_mush: float, w_cact: float, w_dest: float) -> None:

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
        """ Return a cube of each objects presence in the grid

        For each sensor return a numpy matrix with zeros except for the objects
        the sensor is sensitive for. There are n matrices where n is the number
        of sensors. These matrices are combined into a cube.

        Returns:
            numpy 3D array: cube of sensor matrices
        """
        # 
        perceptions = []

        for sensor in self.sensors:
            matrix = sensor.sense_layer()
            perceptions.append(matrix)
        # for 

        cube = np.array(perceptions)
    
        return cube

    ### perceive ###

    def evaluate(self, perceptions: np.array) -> any:
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
        
        # compute desired move
        max_move = 'something'

        return max_move

    ### evaluate ###

### Class: NeuralQ ###


class Q(Vehicle):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)
        
        self.alpha = 0.5       # learning parameter 
        self.gamma = 0.4       # importance of history 
        self.epsilon_max = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99

        
        # Maximal number of steps per game: 10 # gridsize
        self.n_steps = 15 * self.grid.grid_size[0] * self.grid.grid_size[1]   
        self.n_episodes = 2500 # The number of times to complete the maze
        self.actions = []      # Keep track of actions performed when moving
        self.direction = 'X'   # initialize the direction at no move

        self.q_table = np.zeros((self.grid.grid_size[0] * self.grid.grid_size[1], len(ACTIONS)),
                                dtype = float)

    ### __init__ ###

    def get_q_table_spot(self, position):
        """Krijg de locatie in de Q table die hoort bij de positie in de map"""

        spot = position[0] + position[1] * self.grid.grid_size[1] 

        return spot

    ### get_q_table_spot ###

    def get_q_values(self, position, q_table): 
        """ Krijg de q values die horen bij de positie in de map""" 

        spot = self.get_q_table_spot(position)

        return q_table[spot]

    ### get_q_values ###

    def set_q_value(self, position, action, q_table, new_value):
        """ Zet een nieuwe q waarde in de tabel """
        
        self.get_q_values(position, q_table)[action] = new_value

        return

    ### set_q_value ###

    def calculate_new_q_value(self, previous_q_value, reward, max_next_q_value):

        return previous_q_value + self.alpha * ( reward + self.gamma * max_next_q_value - previous_q_value) 

    ### calculate_new_q_value ###

    def show_map(self): # my_map, location):
        """Laat de map met de huidige locatie in de map zien"""

        logger.info(self.grid.print_grid(self.grid.grid_cells))

        return

    ### show_map ###

    def create_q_table(self, filename: str = None):
        # reset the actions of the vehicle
        self.actions = []

        # Create empty Q table
        self.q_table = np.zeros((self.grid.grid_size[0] * self.grid.grid_size[1], len(ACTIONS)),
                                dtype = float)

        # store some statistics
        episodes = []
        episode_lengths = []
        rewards = []
        changes = []

        # backup grid.grid_cells
        backup = np.copy(self.grid.grid_cells)

        # Setting up everything
        epsilon = self.epsilon_max
        runs = 0
        change = 0
        start_location = self.grid.start.location
        destination = self.grid.destination.location

        # perform all the rounds
        for episode in range(self.n_episodes):
            # restore backup
            self.grid.grid_cells = np.copy(backup)

            # Set the location
            self.location = start_location
            location = self.location 
            
            # reset destination
            self.grid.set_destination(Destination, destination)

            # We will go for a linear epsilon greedy approach
            # epsilon = 1 - 0.9 * episode / self.n_episodes
            # Decay the epsilon value every game
            epsilon *= self.epsilon_decay
            if epsilon < self.epsilon_min:
                epsilon = self.epsilon_min
            
            # Perform the steps
            step = 0
            reward_sum = 0
            delta = 0
            previous = 0
            is_finished = False

            while not is_finished and step < self.n_steps:
                # a check on some silly situation which should not occur
                if self.id not in self.grid.things_by_id.keys():
                    logger.info(f'episode {episode}, run {step} I ({self.id}) disappeared')
                    logger.info(self.grid.print_grid(self.grid.grid_cells))

                #for step in range(self.n_steps):
                step += 1

                # Update run counter
                runs += 1
                
                # Explore or exploit
                exploitation_exploration_value = random.uniform(0, 1)

                # exploit
                if exploitation_exploration_value > epsilon:
                    current_q_values = self.get_q_values(location, self.q_table)
                    action = np.argmax(current_q_values)

                # explore
                else:
                    action = random.randint(0, 3)
                
                # Perform action
                self.direction = ACTIONS[action]
                reward, new_location, potential_loc, is_finished = self.move(self.grid)

                # calculate some statistics
                reward_sum += reward
                change = previous - self.q_table.sum()
                previous = self.q_table.sum()
                delta += change

                # ---- Calculate new Q value ---- #
                
                # Get current Q value
                current_q_values = self.get_q_values(location, self.q_table)
                current_q_value = current_q_values[action]
                
                # Get the value of the new state
                next_q_values = self.get_q_values(potential_loc, self.q_table)
                max_next_q_value = np.max(next_q_values)

                # Calculate the new Q value
                new_q_value = self.calculate_new_q_value(current_q_value, reward, max_next_q_value)
                
                # Update Q value
                self.set_q_value(location, action, self.q_table, new_q_value)
                
                # Update location 
                location = new_location

            # for

            episodes.append(episode)
            rewards.append(reward_sum)
            episode_lengths.append(step)
            changes.append(delta)

            #delta = delta / self.n_steps
            if episode % 100 == 0 and episode > 0:
                logger.info(f'Episode {episode}: rewards={mean(rewards[episode-100:]):.2f}, '
                            f'lengths={mean(episode_lengths[episode-100:]):.2f}, '
                            f'change={mean(changes[episode-100:]):.2f}')
            # if
        # for
        
        # reset self.location and destination
        self.location = start_location
        self.grid.set_destination(Destination, destination)

        # Show the number of runs
        logger.info(f'Q table created in {runs} runs')

        # save when filename is not None
        if filename is not None:
            np.savetxt(filename, self.q_table, delimiter = ';')
            logger.info('Q table saved to ' + filename)

        self.plot_results(rewards, episode_lengths, changes)

        return

    ### create_q_table ###

    def plot_results(self, ep_rewards, ep_lengths, ep_changes, weight=0.9):

        # Smoothing function
        def smooth(scalars, weight):  # Weight between 0 and 1
            # (From https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar)
            last = scalars[0]  # First value in the plot (first timestep)
            smoothed = list()
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
                smoothed.append(smoothed_val)                        # Save it
                last = smoothed_val                                  # Anchor the last smoothed value

            return smoothed

        smooth_rewards = smooth(ep_rewards, weight)
        smooth_lengths = smooth(ep_lengths, weight)
        smooth_changes = smooth(ep_changes, weight)

        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(range(len(smooth_rewards)), smooth_rewards)
        ax1.set_title('Episode Rewards (Smoothed {})'.format(weight))
        ax2.plot(range(len(smooth_lengths)), smooth_lengths)
        ax2.set_title('Episode Lengths (Smoothed {})'.format(weight))
        ax3.plot(range(len(smooth_changes)), smooth_changes)
        ax3.set_title('Episode Changes (Smoothed {})'.format(weight))

        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.5, hspace=0.5)
        f.suptitle('Q-Learning: alpha = {}, gamma = {}, epsilon = {}'.format(self.alpha, self.gamma, 0), fontsize=16)
        
        ax2.set_xlabel('Episode')

        ax1.set_ylabel('Reward (Train)')
        ax2.set_ylabel('Length (Train)')
        ax3.set_ylabel('Change (Train)')
        fig_name = 'Q-Learning_TRAIN_alpha_{}_gamma_{}_epsilon_{}.png'.format(self.alpha, self.gamma, 0)
        
        f.savefig(fig_name, dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='png',
            transparent=False, bbox_inches=None, pad_inches=0.1)

        plt.show()

        return

    ### plot_results ###

    def load_or_create_q_table(self, filename: str = None):
        if os.path.isfile(filename):
            logger.info('Loading ' + filename)
            self.q_table = np.genfromtxt(filename, delimiter=';')
            logger.info('Loaded Q table with shape ' + str(self.q_table.shape))

        else:
            logger.info(f'Creating Q table in {self.n_episodes} episodes')
            self.create_q_table(filename)

        # if

        return

    ### load_or_create_q_table ###

    def q_move(self):
        # Reset position
        destination = self.grid.destination.location
        location = self.location
        start = self.location
    
        # Keep track of the actions we took
        actions = []

        self.show_map()
        #self.grid.list_things()
        # Perform the actions
        for step in range(self.n_steps):
            # Take the best action according to our model (always exploid)
            current_q_values = self.get_q_values(location, self.q_table) # , dimensions)
            action = np.argmax(current_q_values)
            
            # Take action
            self.direction = ACTIONS[action]
            reward, new_location, potential_loc, is_finished = self.move(self.grid) # location, action, sample_map)
            
            # Set new location
            location = new_location
            
            # Add action to action list
            actions.append(ACTIONS[action])
            
            # Check if we are done
            if is_finished:
                break
            
            # if

        # for
                
        logger.info(f'Actions: {str(actions)}')

        self.location = start
        self.grid.set_destination(Destination, destination)

        #self.grid.list_things()
        # Show if we made it
        if step == self.n_steps - 1:
            logger.info('Failed')

        else:
            logger.info('Succes!')

        return

    ### q_move ###

    def one_q_move(self):
        # Take the best action according to our model (always exploid)
        current_q_values = self.get_q_values(self.location, self.q_table)
        action = np.argmax(current_q_values)
        
        # Take action
        self.direction = ACTIONS[action]
        reward, new_location, potential_loc, is_finished = self.move(self.grid) 
        
        # Set new location
        self.location = new_location
        
        # Add action to action list
        self.actions.append(ACTIONS[action])
        
        self.destination_reached = is_finished

        return

    ### one_q_move ###

    def next_turn(self):
        super().next_turn()
        
        self.one_q_move()

        return
    
    ### next_turn ###

### Class: Q ###


class Simple(Vehicle):
    def __init__(self, location: tuple, definitions: pd.DataFrame, grid: Grid):
        super().__init__(location, definitions, grid)
        
        # self.type = 'Vehicle'
        # self.category = self.definitions.loc[self.type, COL_CATEGORY]
        # self.mass = self.definitions.loc[self.type, COL_MASS]
        self.direction = 'X'

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

        def get_signal(perceptions, cat):
            if len(perceptions) < 1:
                return 0

            weighted_signal = 0
            for perception in perceptions:
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
                weighted_signal += self.weights[cat] * signal_strength

                logger.debug(cat, move, val, perception, d, signal_strength, weighted_signal, mass)

            # for

            return weighted_signal

        ### get_signal ###


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
            mass = 0

            # look for all perceived objects
            for cat in perceptions:
                # if there is an object perceived
                if len(perceptions[cat]) > 0:
                    # get the first object (with the most signal strength)
                    perceps = perceptions[cat] # [0]]

                    # add to mass
                    signal = get_signal(perceps, cat)
                    mass += signal

                # if
            # for

            evaluated_moves[move] = mass

            if mass > max_val:
                max_val = mass
                max_move = move

        # for

        logger.debug(str(evaluated_moves))

        return max_move

    ### evaluate ###

### Class: Simple ###

