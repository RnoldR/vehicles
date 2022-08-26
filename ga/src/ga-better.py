# -*- coding: utf-8 -*-
import sys
import time
import random
import pandas as pd
from math import log, sin
from bitstring import BitArray #, BitStream

# Code initialisatie: logging
import logging
#import importlib
#importlib.reload(logging)

# create logger
logger = logging.getLogger('ga')

logger.setLevel(10)

# create file handler which logs even debug messages
fh = logging.FileHandler('ga.log')
fh.setLevel(logging.INFO)

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

# Keys into the dictionary holding the variables
NAME = 'name'
BITLEN = 'bit_length'
BITPOS = 'bit_pos'
MAXVAL = 'max_val'
CATEGORY = 'category'
MIN = 'minimum'
MAX = 'maximum'

# forward declaration for self references inside the class
class GA: pass

class GA(object):
    """
    Class representing one or more variables as a single bitstring.
    Each variable is represented by a dictionary containing its name,
    type, min and max and some other properties.    
    """
    seq: int = 0
    
    def __init__(self, ga: GA=None, bits: BitArray=None):
        """ Initializes a GA.

        Args:
            ga (GA, optional): When not None it is a GA to create a copy from
        """
        GA.seq += 1      # increment global counter
        self.id: int = GA.seq # each GA its unique id
        self.fitness_list: list = []
        self.fitness_dict: dict = {}
        self.dna: dict = {}    # dictionary containing all variables
        self.dna_len: int = 0 
        self.bits: BitArray = bits # bits needed to represent all variables
        
        # copy the ga when present
        if ga is not None:
            # copy variables from ga
            for name in ga.dna:
                self.copy_var(ga.dna[name])
                
            # copy fitness keys from ga
            self.fitness_list = ga.fitness_list
            for key in self.fitness_list:
                self.fitness_dict[key] = ga.fitness_dict[key]
                
            # copy the bitstring from ga
            self.bits = ga.bits
        # if
        
        if bits is not None:
            self.bits = bits
        #if
            
        return
    
    ### __init__ ###
    
    def add_var(self, name: str, bit_len: int, category: str, mn, mx):
        """ Adds a variable to the GA.

        Args:
            name (str): Name of the variable
            bit_len (int): # bits to use for this variable
            category (str): F(loat) or I(nt)
            mn (float or int): Minimum of tghe variable
            mx (float or int): Maximum of the variable
        """
        
        # only accept variable when its name is not present
        if name not in self.dna:
            # compute max_val, which is the precision of the variable representation
            max_val = 2**bit_len
                
            # create the variable information
            variable = {NAME: name,
                        BITPOS: 0,
                        BITLEN: bit_len, 
                        MAXVAL: max_val,
                        CATEGORY: category,
                        MIN: mn,
                        MAX: mx
                       } 
            
            # add the variable to the dictionary
            self.dna[name] = variable
            
            # increase the length of the bitstring 
            self.dna_len += bit_len
            
            # synchronize all necessary changes
            self.update()
            
        return
    
    ### add_var ###
    
    def copy_var(self, var):
        self.add_var(var[NAME], var[BITLEN], var[CATEGORY], var[MIN], var[MAX])
            
        return
    
    ### copy_var ###
    
    def get_var(self, name: str):
        """Return variable <name> as float or int.

        Args:
            name (str): Name of the variable

        Returns:
            float or int: The value of the variable, float or int, depending
                on the value of category ('F' or 'I')
        """
        # fetch the variable
        variable = self.dna[name]
        
        # fetch its location on the dna
        start = variable[BITPOS]
        end = start + variable[BITLEN]
        
        # fetch the variable as an unsigned integer between [0, 2^bitlen>
        var = self.bits[start:end].uint
        
        # rescale to variable min and max
        result = variable[MIN] + (var / variable[MAXVAL]) * (variable[MAX] - variable[MIN])
        
        if variable[CATEGORY] == "I":
            return int(result + 0.5) # return as an integer when category is I
        else:
            return result # else as floating point number
        
    ### get_var ###
    
    def update(self):
        """ Compute the position of each variable on the bit string.
        """
        
        # Set the bit length at zero
        self.dna_len = 0
        for key in self.dna:
            variable = self.dna[key]
            
            # set the position of the variable based on bit length
            variable[BITPOS] = self.dna_len
            
            # increment the bit length
            self.dna_len += variable[BITLEN]
            
        return
    
    ### update ###
    
    def randomize(self):
        """ Create a random bit string.
        """
        # create a bitarray with dna_len zero's
        self.bits = BitArray(bin = '0' * self.dna_len)
        
        # just set half of them randomly to one
        for i in range(len(self.bits)):
            if random.random() >= 0.5:
                self.bits[i] = True
                
        return
    
    ### randomize ###
    
    def mutate(self, chromosome: BitArray, p_mutation: float):
        n = len(chromosome)
        if p_mutation < 1:
            n_mutations = int(p_mutation * n)
        else:
            n_mutations = int(p_mutation)
        
        mutations = random.sample(range(n), n_mutations)
        bit_string = chromosome.copy()
        for mutation in mutations: 
            bit_string[mutation] = not bit_string[mutation]
        
        #logger.debug('Mutations: ' + str(mutations))
        #logger.debug(chromosome.bin)
        #logger.debug(bit_string.bin)
        
        return bit_string
    
    ### mutate ###
    
    def crossover(self, chromosome_1: BitArray, chromosome_2: BitArray,
                  p_crossover: float):
        """
        Creates cross overs between two chromosomes. Two children are
        created by copying the bits from either parent. At each cross over
        point the parent from which the bits are copied are swapped.

        Args:
            chromosome_1 (BitArray): 
            chromosome_2 (BitArray): Bitarray's acting as chromosomes.
                Must be of equal length 
            p_crossover (float|int): When float, must be a probability.
                Number of crossovers (p * length chromosome) will be 
                sampled from range(len(chromosome)). When integer it is just
                the number of cross overs. Will be sampled from
                range(len(chromosome))

        Returns:
            tuple of two bitarrays: tuple of the two children
        """
        
        # Compute the number of cross overs
        n = len(chromosome_1)
        if p_crossover < 1: # probability
            n_crossovers = int(p_crossover * n)
        else: # number of cross overs
            n_crossovers = int(p_crossover)
            
        # Sample the cross overs
        crossovers = random.sample(range(1, n-1), n_crossovers)
        
        # insert 0 as the start cross over
        crossovers.insert(0, 0)
        
        # sort from low to high
        crossovers.sort()
        #logger.debug('Crossovers = ' + str(crossovers))

        # create the children as a copy from one parent
        child_1 = chromosome_1.copy()
        child_2 = chromosome_2.copy()
        
        keep = True
        start = 0
        for i in range(n_crossovers+1):
            if i == n_crossovers:
                end = n
            else:
                end = crossovers[i+1]
            # if
                
            if not keep:
                child_1[start:end] = chromosome_2[start:end]
                child_2[start:end] = chromosome_1[start:end]
            # if
                
            keep = not keep
            start = end
        # for
        
        #logger.debug('chrom 1: ' + str(chromosome_1.bin))
        #logger.debug('chrom 2: ' + str(chromosome_2.bin))
        #logger.debug('child 1: ' + str(child_1.bin))
        #logger.debug('child 2: ' + str(child_2.bin))
        
        return child_1, child_2
    
    ### crossover ###
    
    def mate(self, other: GA, p_mutation: float, p_crossover: float):
        parent_1 = self.bits.copy()
        parent_2 = other.bits.copy()
        #n = len(parent_1)
        
        logger.debug('')
        logger.debug('Parent 1: ' + str(parent_1.bin))
        logger.debug('Parent 2: ' + str(parent_2.bin))
        
        chromosome_1, chromosome_2 = self.crossover(parent_1, parent_2,
                                                    p_crossover)
        chromosome_1 = self.mutate(chromosome_1, p_mutation)
        chromosome_2 = self.mutate(chromosome_2, p_mutation)
        
        child_1 = GA(self, bits=chromosome_1)
        child_2 = GA(self, bits=chromosome_2)
        
        logger.debug('Child 1: ' + str(child_1.bits.bin))
        logger.debug('Child 2: ' + str(child_2.bits.bin))
                
        return [child_1, child_2]
    
    ### mate ###
    
    def show(self):
        logger.info('Bitstring: ' + str(self.bits.bin))
        for key in self.dna:
            variable = self.dna[key]
            start = variable[BITPOS]
            end = start + variable[BITLEN]
            logger.info(str(variable[NAME] + ': ' + self.bits[start:end].bin) + 
                        ' = ' + str(self.get_var(variable[NAME])))
            
        return
    
    ### mate ###
    
### Class: GA ###
    

class GaData(object):
    """
    Data for GA
    """
    def __init__(self, X_train=None, X_val=None, X_test=None, 
                 y_train=None, y_val=None, y_test=None):
        self.set_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
        self.data_dict = {}
        
        return
    
    ### __init__ ###
    
    def set_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        return
    
    ### set_datasets ###
    
    def register_variable(self, name: str, value):
        self.data_dict[name] = value
        
        return
    
    ### register_variable ###
    
### Class: Data ###
    
    
class Population(object):
    def __init__(self, p_mutation=0.1, p_crossover=1, fitness=[], 
                 selection_key: str='cpu'):
        
        self.data = GaData()
        self.size = 0
        self.template = GA()
        self.population = []
        self.ff = None
        self.fitness_function = None
        self.fitness_data = None
        self.locals = None
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.selection_key = selection_key
        self.template.fitness_list = fitness
        self.template.fitness_list.insert(0, 'cpu')
        self.template.fitness_dict = {}
        
        for key in self.template.fitness_list:
            self.template.fitness_dict[key] = 0
        
        return
    
    ### __init__ ###
    
    def add_var(self, name, bit_len, category, mn, mx):
        self.template.add_var(name, bit_len, category, mn, mx)
        
        return
    
    ### add_var ###

    def create_population(self, size):
        for i in range(size):
            ga = GA(self.template)
            ga.randomize()
            self.population.append(ga)
            
        return
    
    ### create_population ###
            
    def next_generation(self, selection_size: int):
        """
        Mates all GA's with each other and creates a new
        population based on the offspring. 
        
        Args:
            selection_size (int): Number of best fitted individuals 
                to be selected
        """
        new_population = []
        
        # iterate over all Ga's
        for ga_1 in self.population:
            
            # iterate over all GA's
            for ga_2 in self.population:
                
                # when both GA's are not identical, mate and get their children
                if ga_1.id != ga_2.id:
                    children = ga_1.mate(ga_2, self.p_mutation, self.p_crossover)
                    for child in children:
                        new_population.append(child)
                    # for
                # if
            # for
        # for
        
        # get a selection of the new population based on population parameters
        self.population = self.select(new_population, selection_size, 
                                      self.selection_key)
        
        return
    
    ### next_generation ###
    
    def pre_compute(self):
        '''
        When a population is initially created the fitness of each 
        GA is noit yet known. All fitness is listed as 0 when show()
        is called. pre_compute computes the fitness of each GA in 
        the population. 
        '''
        self.get_fitnesses(self.population)
        
        return
    ### compute_fitnesses ###
            
    def get_fitnesses(self, new_population, selection_key: str=None):
        # build a dictionary of ga.id and fitness
        fitnesses = {}
        for ga in new_population:
            cpu = time.time()
            if self.ff is not None:
                fitness = self.eval_fitness_string(ga)
            elif self.fitness_function is not None:
                fitness = self.eval_fitness_function(ga)
            else:
                raise ValueError('get_fitnesses: ff and fitnes_function are None')
            # if
            
            fitness.insert(0, time.time() - cpu)
            
            for idx, key in enumerate(ga.fitness_list):
                ga.fitness_dict[key] = fitness[idx]
            # for
                
            fitnesses[ga.id] = fitness
        # for
        
        # if no selection_key, return None
        if selection_key is None:
            df = None
        
        # else create dataframe, sort on selection_key and return it
        else:
            
            # create a pandas dataframe from dictionary
            df = pd.DataFrame.from_dict(fitnesses, orient='index',
                                        columns = self.template.fitness_list)
            
            # sort the dataframe on the selection_key
            df = df.sort_values(selection_key, ascending=False)
    
            logger.debug("get_fitnesses: Fitness of GA's")
            for index, row in df.iterrows():
                logger.debug(str(index) + ': ' + str(row))
            # for
        # if
                
        return df
    
    ### get_fitnesses ###
    
    def select(self, new_population, selection_size: int, selection_key: str):
        # build a dictionary of new population based on ga.id
        ga_dict = {ga.id: ga for ga in new_population}
        
        # get fitnesses and sort by selection_key
        sorted_fitnesses = self.get_fitnesses(new_population, selection_key)
        
        # sample the first <selection_size> into a new population
        pop = []
        for idx, row in sorted_fitnesses.iterrows():
            ga = ga_dict[idx]
            pop.append(ga)
            if len(pop) >= selection_size:
                break
            
        return pop
    ### select ###
        
    def set_fitness_string(self, fun: str, local_vars={}):
        """
        Set the fitness function.

        Args:
            fun (str): Fitness function as a string
            local_vars (dict, optional): Dictionary containing local 
                variables referred to in the fitness function
        """
        self.ff = fun
        self.locals = local_vars
        self.fitness_function = None
        
        return
    
    def eval_fitness_string(self, ga: GA):
        """
        Evaluate the fitness function with variables from the GA.

        Args:
            ga (GA): GA to compute the function of

        Returns:
            float: the result of the fitness function
        """
        # copy the dictionary of local variables
        local_vars = self.locals.copy()
        
        # add the variables of the ga
        for key in ga.dna:
            local_vars[key] = ga.get_var(key)
                
        # evaluate the fitness function
        fitness = eval(self.ff, None, local_vars)
        
        return [fitness]
    
    def set_fitness_function(self, function, data):
        self.fitness_function = function
        self.fitness_data = data
        self.ff = None
        
        return
    
    def eval_fitness_function(self, ga: GA):
        # set the variables from ga into the data
        for var in ga.dna:
            value = ga.get_var(var)
            self.fitness_data.data_dict[var] = value
            
        # compute the fitness
        fitness = self.fitness_function(self.fitness_data)
        
        return fitness
    
    def find_ga_id_in_populations(self, id: int):
        for ga in self.population:
            if ga.id == id:
                return ga
            # if
        # for
        
        return None
    ### find_ga_id_in_populations ###
    
    def show(self):
        if len(self.population) == 0:
            logger.warn('Empty population, nothing to show')
            return
        
        # get the fitnesses to show
        #df = self.get_fitnesses(self.population, self.selection_key)
        #df = {x[0]: x[1] for x in fitnesses}
        fit_dict = {}
        for ga in self.population:
            l = []
            for key in self.template.fitness_list:
                l.append(ga.fitness_dict[key])
                
            fit_dict[ga.id] = l
            
        df = pd.DataFrame.from_dict(fit_dict, orient='index', 
                                    columns=self.template.fitness_list)
        
        # first build a format string (fs) and a header string (hs)
        # go over all variables of the first individual
        fs = ['{:5d}:']
        hs = '   ID:'
        
        for key in self.template.fitness_list:
            max = 8
            fs.append(' {:' + str(max) + '.4f}')
            hs += ' '
            if len(key) > max:
                hs += key[-max:]
            else:
                hs += key.rjust(max)
            # if
        # for
        hs += '  '
        
        first = self.population[0]
        length = len(first.bits)
        fs.append('  {:' + str(length) + 's} ')
        hs += 'bits'.rjust(length) + ' '
        
        for key in first.dna:
            variable = first.dna[key]
            min = abs(variable[MIN])
            max = abs(variable[MAX])
            if min > max:
                max = min
            if max <= 0:
                max = 2
            else:
                max = int(log(max)) + 2
            # if
                
            cat = variable[CATEGORY]
            if cat == 'I':
                fs.append('{:' + str(max) + 'd}')
            else:
                max += 3
                fs.append('{:' + str(max) + '.2f}')
            # if
               
            if len(key) > max:
                hs += key[-max:]
            else:
                hs += key.rjust(max)
            # if
        # for
        
        logger.info(hs)
        for idx, row in df.iterrows():
            item = 0
            individual = self.find_ga_id_in_populations(idx)
            line = fs[item].format(individual.id)
            
            for key in df.columns:
                item += 1
                fit_var = row[key]
                line += fs[item].format(fit_var)
            
            item += 1
            line += fs[item].format(individual.bits.bin)
            
            for key in individual.dna:
                item += 1
                var = individual.get_var(key)
                line += fs[item].format(var)
            # for
            
            logger.info(line)
        # for

        return
    
    ### show ###
        
### Class: Population ###

if __name__ == "__main__":
    random.seed(42)

    equation = 'y*sin(x)'
    pop = Population(p_mutation=0.2, p_crossover=2, 
                     fitness=[equation], 
                     selection_key=equation)
    pop.add_var('x', 16, 'F', 0.0, 6.28)    
    pop.add_var('y', 2, 'I', 1, 4)
    pop.set_fitness_string(equation)
    pop.create_population(10)
    
    logger.info('--- Generation 0 ---')
    
    # pre-compute the fitnesses for the first time because that has
    # not yet been computed and we still want to see it in show()
    pop.get_fitnesses(pop.population, equation)
    pop.show()

    for generation in range(1, 3):
        logger.info('')
        logger.info('*** Generation ' + str(generation))
        pop.next_generation(10)
        pop.show()
    