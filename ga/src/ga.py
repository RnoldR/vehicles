import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info('Initializing GA')

from create_log import create_logger

import sys
import time
import random
import numpy as np
import pandas as pd
from math import log10
from bitstring import BitArray #, BitStream

# raise an exception when a numpy error occurs
#np.seterr(all='raise')

# Keys into the dictionary holding the variables
NAME = 'name'
BITLEN = 'bit_length'
BITPOS = 'bit_pos'
MAXVAL = 'max_val'
LENGTH = 'array_length'
MAXLENGTH = 'max_length'
FORMATSTRING = 'format_string'
FORMATHEADER = 'format_header'
CATEGORY = 'category'
TYPE = 'variable_type'
MIN = 'minimum'
MAX = 'maximum'
EXTRA = 'extra'
LIST = 'list'
LISTCAT = 'list_category'

TYPE_SCALAR = 'scalar'
TYPE_ARRAY = 'array'
TYPE_VAR_ARRAY = 'variable array'

"""
integers are always in the range 0..2^BITLEN. That is not always 
desired when the maximum is not an exact power of 2. When setiing
the value of the variable MIN is subtracted and when getting the 
value MIN is added. 
How to handle the extra values between MAX and 2^BITLEN
is handled by HANDLE_SUPERFLUOUS. Its values are:
   EX_IGNORE - just accept the larger than MAX values
   EX_MOD    - the values are modded, side-effect is that some 
               integers occur more frequently than others
   EX_RANDOM - randomize the values between MIN and MAX
"""

EX_IGNORE = 'ignore'
EX_RANDOM = 'random'
EX_MOD = 'mod'

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
        """ 
        Initializes a GA.

        Args:
            ga (GA, optional): When not None it is a GA to create a copy from
            bits (bitstring, optional): initializes bitstring when not None
        """
        GA.seq += 1      # increment global counter
        self.id: int = GA.seq # each GA its unique id
        self.fitness_list: list = []
        self.fitness_dict: dict = {}
        self.dna: dict = {}    # dictionary containing all variables
        self.dna_len: int = 0 
        self.bits: BitArray = bits # bits needed to represent all variables
        self.generation = 0

        # number of GA's rejected because of dead on arrival
        self.rejects_doa: int = 0
        self.rejects_doa_list: list = []

        # number of GA's rejected because of unviablility
        self.rejects_unviability: int = 0
        self.rejects_unviability_list: list = []

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
        # if
            
        return
    
    ### __init__ ###


    def __str__(self):
        return (f'{self.id}: rejects = {self.rejects_doa}, unviable = {self.rejects_unviability} '\
                f'dna: {str(self.dna)}')

    ### __str__ ###

    
    def add_var(self, 
                name: str, 
                bit_len: int, 
                category: str,
                mn, mx, 
                var_type: str = TYPE_SCALAR,
                length: int=1,
                max_length: int=None,
                extra: str = None,
                value_list = None,
               ):
        """
        Adds a variable to the GA.

        Args:
            name (str): Name of the variable
            bit_len (int): # bits to use for this variable
            category (str): F(loat) or I(nt)
            mn (float or int): Minimum of the variable
            mx (float or int): Maximum of the variable
            var_type (str): type of variable (scalar or array)
            length (int): length of the array, 1 for scalar
        """

        def format_create(var_name, min, max, cat):
            if max <= 0:
                max = 5

            else:
                max = int(log10(max))

            if max < 5:
                max = 5

            if cat == 'I':
                format_string = '{:' + str(max) + 'd} '

            elif cat == 'S':
                format_string = '{:' + str(max) + 's} '

            elif cat == 'F':
                max += 3
                format_string = '{:' + str(max) + '.2f} '

            else:
                raise ValueError(f'Invalid catagorye: {cat} for variable "{var_name}"')

            # if

            if len(name) > max:
                header_string = name[:max] + ' '

            else:
                header_string = name.rjust(max) + ' '

            return format_string, header_string

        ### format_create ###

        def format_list(var_name, values):
            list_cat = 'I'
            format_length = 0

            for val in values:
                if isinstance(val, int):
                    value = len(str(val))

                elif isinstance(val, float):
                    value = len(f'{val:.2f}')

                    if list_cat != 'S':
                        list_cat = 'F'

                else:
                    value = 10 ** len(str(val))

                    list_cat = 'S'

                if value > format_length:
                    format_length = value

                # if

            # for

            format_string, header_string = format_create(var_name, 0, format_length, list_cat)

            return format_string, header_string, list_cat

        ### format_list ###

        list_cat = None

        # a name should be specified
        if len(name) < 1:
            raise ValueError(f'A variable name must be specified: "{name}"')

        # bit_len must by 1 or more
        if bit_len < 1:
            raise ValueError(f'bit_len of array variable "{name}" should be at least 1')

        # max should be larger than min
        if mx <= mn:
            raise ValueError(f'maximum variable "{name}" should be larger than min')

        if isinstance(length, str) and var_type == TYPE_ARRAY:
            var_type = TYPE_VAR_ARRAY

        if extra is None:
            extra = EX_MOD
            
        if var_type == TYPE_SCALAR:
            if length != 1:
                logger.warning(f'Scalar "{name}" must have length 1 instead of {length}. Assumed')

            max_length = 1

            if value_list is None:
                format_string, header_string = format_create(name, mn, mx, category)

            else:
                category = 'I'
                mn = 0
                mx = len(value_list) - 1

                format_string, header_string, list_cat = format_list(name, value_list)
                bit_len = mx.bit_length()

        elif var_type == TYPE_ARRAY:
            if not isinstance(length, int):
                raise ValueError(f'Array variable "{name}" must have an integer length: {length}')

            if length < 1:
                raise ValueError(f'Length of array variable "{name}" should be at least 1 instead of {length}')

            max_length = length

            if value_list is None:
                format_string, header_string = format_create(name, mn, mx, category)

            else:
                category = 'I'
                mn = 0
                mx = len(value_list) - 1

                format_string, header_string, list_cat = format_list(name, value_list)
                bit_len = mx.bit_length()

                
        elif var_type == TYPE_VAR_ARRAY:
            if isinstance(length, str):
                # length is a string, check its occurrence in dna
                if length in self.dna:
                    variable = self.dna[length]

                    if round(variable[MIN]) < 0:
                        raise ValueError(f'Minimum length for variable array "{name}" length '
                                         f'variable "{variable[NAME]} should be zero or larger')
                    
                    if max_length is None:
                        max_length = round(variable[MAX])

            if max_length is None or max_length < 1:
                raise ValueError(f'max_length for "{name}" should be specified and > 0 for variable array')

            if value_list is None:
                format_string, header_string = format_create(name, mn, mx, category)

            else:
                category = 'I'
                mn = 0
                mx = len(value_list) - 1

                format_string, header_string, list_cat = format_list(name, value_list)
                bit_len = mx.bit_length()


        else:
            raise ValueError (f'Illegal variable type for variable "{name}": {var_type}')

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
                        MAX: mx,
                        TYPE: var_type,
                        LENGTH: length, 
                        MAXLENGTH: max_length,
                        FORMATSTRING: format_string,
                        FORMATHEADER: header_string,
                        EXTRA: extra,
                        LIST: value_list,
                        LISTCAT: list_cat,
                       } 
            
            # add the variable to the dictionary
            self.dna[name] = variable
            
            # increase the length of the bitstring 
            self.dna_len += max_length * bit_len
            
            # synchronize all necessary changes
            self.update()

            if not isinstance(length, int):
                logger.debug(str(variable))

        # Variable name alreay exists, raise exception
        else:
            raise ValueError(f'Variable name {name} already exists')
            
        return
    
    ### add_var ###
    
    
    def copy_var(self, var):
        value_list = None if var[LIST] is None else var[LIST].copy()
        self.add_var(var[NAME], var[BITLEN], var[CATEGORY], var[MIN], var[MAX], var[TYPE], var[LENGTH],
                     var[MAXLENGTH], var[EXTRA], value_list)
            
        return
    
    ### copy_var ###

    
    def set_var_int(self, name: str, value: int):
        # fetch the variable
        variable = self.dna[name]
        
        # perform checks
        if value < variable[MIN] or value >= variable[MAX]:
            raise ValueError(f'value {value:d} should be >= {variable[MIN]:d} or < {variable[MAX]:d}')
        
        # fetch its location on the dna
        start = variable[BITPOS]
        end = start + variable[BITLEN]
        
        value = value - variable[MIN]
        value_bits = "{num:0{width}b}".format(num=value, width=variable[BITLEN])
        
        self.bits = self.bits[:start] + BitArray(bin=value_bits) + self.bits[end:]
         
        return
        
    ### set_var_int ###


    def set_var_float(self, name: str, value: int):
        # fetch the variable
        variable = self.dna[name]
        
        # perform checks
        if value < variable[MIN] or value >= variable[MAX]:
            raise ValueError(f'value should be >= {variable[MIN]:f} or < {variable[MAX]:f}')
        
        # fetch its location on the dna
        start = variable[BITPOS]
        end = start + variable[BITLEN]
        
        range = 2 ** variable[BITLEN] - 1
        value = int((value - variable[MIN]) / (variable[MAX] - variable[MIN]) * range)
        value_bits = "{num:0{width}b}".format(num=value, width=variable[BITLEN])
        
        self.bits = self.bits[:start] + BitArray(bin=value_bits) + self.bits[end:]
         
        return
        
    ### set_var_float ###


    def set_var_int_array(self, name: str, value_list: list):
        # fetch the variable
        variable = self.dna[name]
        
        for idx in range(variable[LENGTH]):
            value = value_list[idx]

            # perform checks
            if value < variable[MIN] or value >= variable[MAX]:
                raise ValueError(f'value {value} should be >= {variable[MIN]} or < {variable[MAX]}')
            
            # fetch its location on the dna
            start = variable[BITPOS] + idx * variable[BITLEN]
            end = start + variable[BITLEN]
            
            value = value - variable[MIN]
            value_bits = "{num:0{width}b}".format(num=value, width=variable[BITLEN])
            
            self.bits = self.bits[:start] + BitArray(bin=value_bits) + self.bits[end:]

        # for
         
        return
        
    ### set_var_int_array ###


    def get_var_bits(self, name: str):
        """ Return variable <name> as float or int.

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
        var = self.bits[start:end]
        
        return var

    ### get_var_bits ###


    def get_array_length(self, name: str):
        # fetch the variable
        variable = self.dna[name]

        if isinstance(variable[LENGTH], int):
            return variable[LENGTH]

        else:
            value = self.get_var(variable[LENGTH])

            return round(value)

        #if

    ### get_array_length ###


    def get_scalar_var(self, name: str, index: int = 0):
        """ Return variable <name> as float or int.

        Args:
            name (str): Name of the variable

        Returns:
            float or int: The value of the variable, float or int, depending
                on the value of category ('F' or 'I')
        """
        # fetch the variable
        variable = self.dna[name]

        # check array index
        if variable[TYPE] in [TYPE_SCALAR]:
            if index != 0:
                raise ValueError(f'Index of scalar or list "{name}" should be zero')

        elif variable[TYPE] in [TYPE_ARRAY, TYPE_VAR_ARRAY]:
            if index < 0 or index >= variable[MAXLENGTH]:
                raise ValueError(f'Index of array {name} should be range [0-{variable[MAXLENGTH]}], is: {index}')

        else:
            raise ValueError(f'Unknown type ({variable[TYPE]} for variable "{name}"')

        # if
        
        # fetch its location on the dna
        start = variable[BITPOS] + index * variable[BITLEN]
        end = start + variable[BITLEN]
        
        # fetch the variable as an unsigned integer between [0, 2^bitlen>
        var = self.bits[start:end].uint

        category = variable[CATEGORY]

        # for lists the category is always I, regardless of real content
        if variable[LIST] is not None:
            category = 'I'
        
        if category == "F":
            # rescale to variable min and max
            result = variable[MIN] + (var / variable[MAXVAL]) * (variable[MAX] - variable[MIN])
            
            return result
        
        elif category == "I":
            # return as an integer when category is I

            # look on how to handle above max cases, first rescale MAX
            max = variable[MAX] - variable[MIN] + 1
            if var >= max:
                if variable[EXTRA] == EX_IGNORE:
                    pass

                elif variable[EXTRA] == EX_MOD:
                    var = var % max

                elif variable[EXTRA] == EX_RANDOM:
                    var = random.randint(0, max)

                else:
                    raise ValueError(f'Illegal variable[EXTRA] option: "{variable[EXTRA]}"')
                # if

            if variable[LIST] is None:
                result = var + variable[MIN]
            else:
                result = variable[LIST][var]
            
            return result
        
        else:
            raise ValueError(f'Unknown category: "' + variable[CATEGORY] + '"')
            
            return None # else as floating point number
        
    ### get_scalar_var ###
    
    
    def get_array_var(self, name: str):
        """ Returns array variable <name> as a list of float or int.

        Args:
            name (str): Name of the variable

        Returns:
            list of float or int: The value of the variable, float or int, depending
                on the value of category ('F' or 'I')
        """

        values = []
        n = self.get_array_length(name)
        for i in range(n):
            val = self.get_scalar_var(name, index=i)            
            values.append(val)
        # for

        return values

    ### get_array_var ###


    def get_var(self, name: str, index: int = None):
        variable = self.dna[name]
        
        if variable[TYPE] == TYPE_SCALAR:
            result = self.get_scalar_var(name)

        elif variable[TYPE] in [TYPE_ARRAY, TYPE_VAR_ARRAY]:
            result = self.get_array_var(name)
            
            if index is not None:
                result = result[index]

            # if

        else:
            logger.critical(f'*** Unknown variable type {variable[TYPE]} for variable "{name}" ***')

        # if

        return result

    ### get_var ###

    
    def update(self):
        """
        Compute the position of each variable on the bit string.
        """
        # Set the bit length at zero
        self.dna_len = 0
        for key in self.dna:
            variable = self.dna[key]
            
            # set the position of the variable based on bit length
            variable[BITPOS] = self.dna_len
            
            # increment the bit length
            self.dna_len += variable[BITLEN] * variable[MAXLENGTH]
            
        return
    
    ### update ###

    
    def zeros(self):
        s = self.dna_len * '0'
        
        self.bits = BitArray(bin=s)

        return s
    
    ### zeros ###
    
    
    def ones(self):
        s = self.dna_len * '1'
        
        self.bits = BitArray(bin=s)

        return s
    
    ### ones ###
    
    
    def randomize(self):
        """
        Create a random bit string.
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
        
        logger.debug('Mutations: ' + str(mutations))
        logger.debug(str(chromosome.bin))
        logger.debug(str(bit_string.bin))
        
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
        logger.debug('Crossovers = ' + str(crossovers))

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
        
        logger.debug('chrom 1: ' + str(chromosome_1.bin))
        logger.debug('chrom 2: ' + str(chromosome_2.bin))
        logger.debug('child 1: ' + str(child_1.bin))
        logger.debug('child 2: ' + str(child_2.bin))
        
        return child_1, child_2
    
    ### crossover ###
    
    
    def mate(self, other: GA, p_mutation: float, p_crossover: float):
        np.seterr(all='raise')

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
        logger.info(f'')
        logger.info(f'=== Individual ID = {self.id} ===')
        logger.info('Bitstring: ' + str(self.bits.bin))

        for key in self.dna:
            variable = self.dna[key]
            min = variable[MIN]
            max = variable[MAX]
            cat = variable[CATEGORY]
            bits = self.get_var_bits(key)
            
            if variable[TYPE] == TYPE_SCALAR:
                val = self.get_var(key)

                start = variable[BITPOS]
                end = start + variable[BITLEN]
                bits = self.bits[start:end].bin            
                logger.info(f'{key}: {bits} - {val} ({cat}) [{min}-{max}]')

            elif variable[TYPE] == TYPE_ARRAY:
                logger.info(f'{key}: {bits} - (Array of {cat}, {self.get_array_length(key)} elements) [{min}-{max}]')

                for idx in range(self.get_array_length(key)):
                    val = self.get_var(key, index=idx)
                    start = variable[BITPOS] + idx * variable[BITLEN]
                    end = start + variable[BITLEN]
                    bits = self.bits[start:end].bin

                    logger.info(f'  {idx}: {bits} - {val}')

                # for

            elif variable[TYPE] == TYPE_VAR_ARRAY:
                len_name = variable[LENGTH]
                len_val = self.get_array_length(key)
                logger.info(f'{key}: {bits} - (Variable Array of {cat}, {len_name} = {len_val} elements) [{min}-{max}]')

                for idx in range(len_val):
                    val = self.get_var(key, index=idx)
                    start = variable[BITPOS] + idx * variable[BITLEN]
                    end = start + variable[BITLEN]
                    bits = self.bits[start:end].bin

                    logger.info(f'  {idx}: {bits} - {val}')

                # for

            # if

            logger.info('')

        # for
            
        return
    
    ### show ###
    
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


class Criterion():
    def __init__(self, key, comparison, value):
        self.selection_key = key
        self.selection_comp = comparison
        self.selection_value = value
        
        return
        
    ### __init__ ###
    
### Class: Criterion ###
    
    
class Population(object):

    generation_seq: int = 0

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
        self.template.fitness_list = [x for x in fitness]
        #self.template.fitness_list = [x[0] for x in fitness]
        #self.template.fitness_list.insert(0, 'cpu')

        self.template.fitness_dict = {}
        for key in self.template.fitness_list:
            self.template.fitness_dict[key] = 0

        self.n_best_of: int = 0
        self.the_best_of = {}
        
        return
    
    ### __init__ ###
    
    def add_var_int(self, name: str, mn: int=None, mx: int=None, value_list=None):
        if value_list is None:
            if mn is None or mx is None:
                raise ValueError(f'Minimum and maximum should be supplied for variable "{name}"')
        else:
            mn = 0
            mx = len(value_list)

        # if

        range = mx - mn
        self.template.add_var(name, range.bit_length(), 'I', mn, mx, value_list=value_list)
        
        return
    
    ### add_var_int ###
    

    def add_var_float(self, name: str, bit_len: int, mn: float, mx: float):
        self.template.add_var(name, bit_len, 'F', mn, mx)
        
        return
    
    ### def_var_float ###
    
    
    def add_var_int_array(self, name: str, mn: int=None, mx: int=None, length: int=None, value_list=None):
        if value_list is None:
            if mn is None or mx is None:
                raise ValueError(f'Minimum and maximum should be supplied for in array "{name}"')
        else:
            mn = 0
            mx = len(value_list)

        # if

        range = mx - mn
        if length == None:
            raise ValueError('No length specified for int array "{name}"')

        self.template.add_var(name, range.bit_length(), 'I', mn, mx, TYPE_ARRAY, length, value_list=value_list)
        
        return
    
    ### add_var_int ###
    

    def add_var_float_array(self, name: str, bit_len: int, mn: float, mx: float, length: int):
        self.template.add_var(name, bit_len, 'F', mn, mx, TYPE_ARRAY, length)
        
        return
    
    ### add_var_float_array ###


    def set_var_int(self, name: str, value: int):
        self.template.set_var_int(name, value)
        
        return
    
    ### set_var_int ###
    
    
    def set_var_float(self, name: str, value: float):
        self.template.set_var_float(name, value)
        
        return
    
    ### set_var_float ###
    

    def set_var_int_array(self, name: str, value_list: list):
        self.template.set_var_int_array(name, value_list)
        
        return
    
    ### set_var_int_array ###
    
    
    def create_population(self, size: int):
        self.generation_seq = 0

        for i in range(size):
            ga = GA(self.template)
            ga.randomize()
            ga.generation = self.generation_seq
            self.population.append(ga)
            
        return
    
    ### create_population ###
    
    
    def create_randomized_child(self):
        children = []
        
        # as long as there are no children
        while len(children) == 0:
            # sample parents from the current population
            parents = random.sample(self.population, 2)
            children = parents[0].mate(parents[1], self.p_mutation, self.p_crossover)
            
        # while
        
        # There are children, return them
        return children
    
    ### create_randomized_child ###
        
            
    def next_generation(self, selection_size: int, crit: Criterion):
        """
        Mates all GA's with each other and creates a new
        population based on the offspring. 
        
        Args:
            selection_size (int): Number of best fitted individuals 
                to be selected
        """

        self.generation_seq += 1

        np.seterr(all='raise')
        new_population: list = []

        # iterate over all Ga's
        for ga_1 in self.population:
            
            # iterate over all GA's
            for ga_2 in self.population:
                # when both GA's are not identical, mate and get their children
                if ga_1.id != ga_2.id:
                    children = ga_1.mate(ga_2, self.p_mutation, self.p_crossover)

                    for child in children:
                        child.generation = self.generation_seq
                        new_population.append(child)
                    # for
                # if
            # for
        # for
        
        # keep the best n individuals from current population
        if self.n_best_of > 0:
            for i in range(self.n_best_of):
                ga = self.population[i]
                self.the_best_off[ga.id] = ga
        
        # get a selection of the new population based on population parameters
        self.population = self.select(new_population, selection_size, crit)

        return
    
    ### next_generation ###
    
    
    def pre_compute(self, crit: Criterion):
        """
        When a population is initially created the fitness of each 
        GA is not yet known. All fitness is listed as 0 when show()
        is called. pre_compute computes the fitness of each GA in 
        the population. 
        """
        self.get_fitnesses(self.population, crit)
        
        return
    ### compute_fitnesses ###
    
            
    def get_fitnesses(self, new_population, crit: Criterion):
        
        # build a dictionary of ga.id and fitness
        fitnesses = {}
        for ga in new_population:
            cpu = time.time()
            if self.ff is not None:
                fitness = self.eval_fitness_string(ga)

            elif self.fitness_function is not None:
                fitness = self.eval_fitness_function(ga, crit)

            else:
                raise ValueError('get_fitnesses: ff and fitnes_function are None')
            # if
            
            for key in ga.fitness_list:
                ga.fitness_dict[key] = fitness[key]
            # for
                
            fitnesses[ga.id] = fitness
        # for
        
        # if no selection_key, return None
        if crit.selection_key is None:
            df = None
        
        # else create dataframe, sort on selection_key and return it
        else:
            
            # create a pandas dataframe from dictionary
            df = pd.DataFrame.from_dict(fitnesses, orient='index',
                                        columns = self.template.fitness_list)
            
            # sort the dataframe on the selection_key
            if crit.selection_comp == 'ge':
                df = df.sort_values(crit.selection_key, ascending=False)

            elif crit.selection_comp == 'le':
                df = df.sort_values(crit.selection_key, ascending=True)

            elif crit.selection_comp == 'eq':
                df[crit.selection_key] = abs(df[crit.selection_key] - 
                                             crit.selection_value)
                df = df.sort_values(crit.selection_key, ascending=True)

            else:
                error = '*** ga.get_fitnesses: expected "ge", "le" or "eq" ' \
                        f'for select_comparison, got "{crit.selection_comp}".'
                        
                raise ValueError(error)
            # if             
    
            logger.debug("get_fitnesses: Fitness of GA's")
            for index, row in df.iterrows():
                logger.debug(str(index) + ': ' + str(row))
            # for
        # if
                
        return df
    
    ### get_fitnesses ###
    
    
    def select(self, new_population, selection_size: int, crit: Criterion):
        # build a dictionary of new population based on ga.id
        ga_dict = {ga.id: ga for ga in new_population}
        
        # get fitnesses and sort by selection_key
        sorted_fitnesses = self.get_fitnesses(new_population, crit)
        
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
    
    ### set_fitness_string ###
    
    
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
    
    ### eval_fitness_string ###
    
    
    def set_fitness_function(self, function, data):
        """
        Sets the fitness function and data to operate upon.

        Args:
            function (TYPE): function to compute the fitness.
            data (TYPE): data as argument to fitness.

        Returns:
            None.

        """

        self.fitness_function = function
        self.fitness_data = data
        self.ff = None
        
        return
    
    ### set_fitness_function ###
    
    
    def eval_fitness_function(self, ga: GA, crit: Criterion):
        # set the variables from ga into the data
        for var in ga.dna:
            value = ga.get_var(var)
            self.fitness_data.data_dict[var] = value
            
        # compute the fitness
        cpu = time.time()
        fitness = self.fitness_function(self.fitness_data, crit)
        cpu = time.time() - cpu

        # add for each fitness the fitness / cpu time as extra fitness measure
        # also add cpu
        result = {}
        for fit in fitness:
            result[fit] = fitness[fit]
            parts = fit.split('_')
            cpu_part = parts[-1] + '/cpu'
            result[cpu_part] = fitness[fit] / cpu

        result['cpu'] = cpu

        return result
    
    ### eval_fitness_function ###
    
    
    def find_ga_by_id(self, id: int):
        """ finds a GA by its id in the population

        Args:
            id (int): ID of the GA

        Returns:
            GA: GA if ID exists, else None
        """

        for ga in self.population:
            if ga.id == id:
                return ga
            # if
        # for
        
        return None
    
    ### find_ga_by_id ###


    def find_ga_by_seq(self, id: int):
        """ Finds an individual GA in the population b y its sequence number.
            id = 0 is first GA, id = 1 is second GA, etc

        Args:
            id (int): Sequence number in the population

        Returns:
            GA: GA when one is found, else None
        """

        if self.population is None:
            return None

        if id < len(self.population):
            return self.population[id]

        return None
    
    ### find_ga_by_seq ###


    def get_pop_as_df(self):
        """ Converts the population to a Pandas.DataFrame with variable names
            as columns.

        Returns:
            Pandas.DataFrame: Converted population
        """

        if len(self.population) == 0:
            logger.warning('Empty population, nothing to show')
            return None
        
        # get the fitnesses to show
        fit_dict = {}
        for ga in self.population:
            l = []
            for key in self.template.fitness_list:
                l.append(ga.fitness_dict[key])
                
            fit_dict[ga.id] = l
            
        df = pd.DataFrame.from_dict(fit_dict, orient='index', 
                                    columns=self.template.fitness_list)

        
        return df

    ### get_pop_as_df ###


    def show_variables(self):
        for name in self.template.dna:
            var = self.template.dna[name]
            logger.info ('')
            logger.info(f'=== {var[NAME]} ===')

            for key, value in var.items():
                if key != "NAME":
                    logger.info(f'{key}: {value}')

                # if
            # for
        # for

        return

    ### show_variables ###


    def show(self):
        np.seterr(all='raise')
        df = self.get_pop_as_df()
        if df is None:
            raise ValueError('No population to show (df == None)')

        # first build a format string (fs) and a header string (hs)
        # go over all variables of the first individual
        fs = ['{:5d}:']
        hs_1 = '   ID:'
        
        # header and format string for fitness criteria
        for key in self.template.fitness_list:
            fitness_format = ' {:8.4f}'
            fs.append(fitness_format)#' {:' + str(max) + '.4f}')

            hs_1 += ' '
            if len(key) > len(fitness_format):
                hs_1 += key[-len(fitness_format):]
            else:
                hs_1 += key.rjust(len(fitness_format))
            # if
        # for

        # add the bitstring for headers and format
        hs_1 += '  '
        first = self.population[0]
        length = len(first.bits)
        fs.append('  {:' + str(length) + 's} ')
        hs_1 += 'bits'.rjust(length) + ' '
        hs_2 = (len(hs_1)) * ' '
        
        # add the variables
        for key in first.dna:
            variable = first.dna[key]
            cat = variable[CATEGORY]
            min = abs(variable[MIN])
            max = abs(variable[MAX])

            if min > max:
                max = min

            if max <= 0:
                max = 2

            else:
                max = int(log10(max)) + 2

            # if

            # create format string for header
            if variable[TYPE] == TYPE_SCALAR:
                fs.append(variable[FORMATSTRING])
                hs_1 += variable[FORMATHEADER]
                hs_2 += (len(variable[FORMATSTRING])) * ' '

            # create format string for (variable) array
            elif variable[TYPE] in [TYPE_ARRAY, TYPE_VAR_ARRAY]:
                format_string = variable[FORMATSTRING]
                header_string = variable[FORMATHEADER]
                # iterate over all array elements
                for idx in range(variable[MAXLENGTH]):
                    fs.append(format_string)

                    # create the index numbers for header string 1
                    if idx == 0:
                        header_len = len(header_string) - 1
                    else:
                        header_len = len(header_string)

                    header = f'{idx:{header_len}d}'
                    hs_1 += header
                    
                # for
                
                hs_1 += ' '

                # array name for header string 2
                len_max = variable[MAXLENGTH] * len(header_string)

                if len(key) > len_max:
                    hs_2 += key[:len_max] + ' '

                else:
                    hs_2 += key.ljust(len_max) + ' '

                # if

            else:
                raise ValueError(f'Unknown variable type: {variable[TYPE]}')

            # if

        # for
        
        # print the headers
        logger.info(hs_2)
        logger.info(hs_1)

        # print the format strings
        for idx, row in df.iterrows():
            item = 0
            individual = self.find_ga_by_id(idx)
            line = fs[item].format(individual.id)
            
            for key in df.columns:
                item += 1
                fit_var = row[key]
                line += fs[item].format(fit_var)
            # for
            
            item += 1
            line += fs[item].format(individual.bits.bin)
            
            for key in individual.dna:
                variable = individual.dna[key]

                for idx in range(variable[MAXLENGTH]):
                    item += 1

                    try:
                        var = individual.get_var(key, index=idx)
                    except:
                        var = -1
                    # try..except

                    formatted_var = fs[item].format(var)

                    if variable[TYPE] != TYPE_VAR_ARRAY:
                        line += formatted_var
                    else:
                        array_len = round(individual.get_var(variable[LENGTH]))
                        if idx < array_len:
                            line += formatted_var

                        else:
                            line += (len(formatted_var) - 2) * ' ' + '-' + ' '

                        # if
                    # if
                # for
            # for
            
            logger.info(line)
        # for

        return
    
    ### show ###
    
        
### Class: Population ###


def ga_simple_test():
    logger.info('')
    logger.info('*** ga_simple_test ***')

    fitnesses = ['cpu', 'val_f1', 'f1/cpu']

    pop = Population(p_mutation=0.25, 
                     p_crossover=5, 
                     fitness=fitnesses, 
                     selection_key=fitnesses[1])

    data = GaData()
    data.register_variable('verbose', 1)
    data.register_variable('n_epochs', 10)
    data.register_variable('batch_size', 128)
    data.register_variable('n_layers', 4)
    data.register_variable('real', 3.14)

    crit = Criterion('val_f1', 'ge', 1.0)

    pop.add_var_int('n_epochs', 10, 73)
    pop.add_var_int('n_layers', 3, 7)
    pop.add_var_float('real', 10, 1.5, 4.2)

    pop.create_population(4)

    pop.show_variables()
    pop.show()

    print(pop.size)

    for gene in pop.population:
        gene.show()

    return

### ga_simple_test ###


def ga_simple_array_test():
    logger.info('')
    logger.info('*** ga_simple_array_test ***')
    
    fitnesses = ['val_acc', 'val_f1', 'acc/cpu']

    pop = Population(p_mutation=0.25, 
                     p_crossover=5, 
                     fitness=fitnesses, 
                     selection_key=fitnesses[1])

    data = GaData()

    n_layers = 3

    # resgister variable before you can use them
    data.register_variable('n_layers', n_layers)
    data.register_variable('Example_coefficients', 0)
    data.register_variable('epochs', 0)
    data.register_variable('floats', 0)
    data.register_variable('bsize', 0)

    crit = Criterion('val_f1', 'ge', 1.0)

    # now add the variables you want to use in your GA
    pop.add_var_int('n_layers', 3, 7)
    pop.add_var_int_array('Example_coefficients', 0, 15, 'n_layers')
    pop.add_var_int('epochs', 10, 100)
    pop.add_var_float('bsize', 6, 0, 3.14)

    pop.create_population(4)

    pop.show_variables()
    pop.show()

    for gene in pop.population:
        gene.show()

    individual = pop.find_ga_by_seq(1)

    var = individual.get_var('n_layers')
    logger.info(f'n_layers: {var}')

    var = individual.get_var('Example_coefficients')
    logger.info(f'Example coefficients: {var}')

    var = individual.get_var('Example_coefficients', index=2)
    logger.info(f'Example_coefficients[2]: {var}')

    try:
        var = individual.get_var('Example_coefficients', index=5)
        logger.info(f'Example_coefficients[5]: {var}')

    except:
        logger.info('Expected crash')

    return

### ga_simple_array_test ###


def ga_simple_list_test():
    logger.info('')
    logger.info('*** ga_simple_list_test ***')
    
    fitnesses = ['val_acc', 'val_f1', 'acc/cpu']
    layer_sizes = [32, 64, 128, 256, 512, 1024]
    kernel_sizes = [3, 5, 7]

    pop = Population(p_mutation=0.25, 
                     p_crossover=5, 
                     fitness=fitnesses, 
                     selection_key=fitnesses[1])

    data = GaData()

    # resgister variable before you can use them
    data.register_variable('n_layers', 2)
    data.register_variable('fitnesses', fitnesses)
    data.register_variable('layers', layer_sizes)
    data.register_variable('kernels', kernel_sizes)

    crit = Criterion('val_f1', 'ge', 1.0)

    # now add the variables you want to use in your GA
    pop.add_var_int('n_layers', 2, 4)
    pop.add_var_int('fitnesses', value_list=fitnesses)
    pop.add_var_int_array('kernels', length='n_layers', value_list=kernel_sizes)
    pop.add_var_int_array('layers', length='n_layers', value_list=layer_sizes)

    pop.create_population(4)

    pop.show_variables()
    pop.show()

    for gene in pop.population:
        gene.show()

    individual = pop.find_ga_by_seq(1)

    var = individual.get_var('fitnesses')
    logger.info(f'fitness: {var}')

    var = individual.get_var('layers')
    logger.info(f'layer_size: {var}')

    var = individual.get_var('kernels')
    logger.info(f'kernel_size: {var}')

    return

### ga_simple_list_test ###


def ga_demo_linear_regression():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from math import sqrt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def fitness_linreg(data: GaData, crit: Criterion):
        # fetch the ML data
        X_train = data.X_train
        X_val = data.X_val
        y_train = data.y_train
        y_val = data.y_val

        # fetch the parameters for the logistic regression from data
        a1 = data.data_dict['a1']
        a2 = data.data_dict['a2']
        
        # Create a logistic regression classifier
        #classifier = LinearRegression()
        #classifier.fit(X_train, y_train)
        
        y_pred = a1 * X_val + a2 # classifier.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = 0
        if mse > 0:
            rmse = sqrt(mse)

        return {'val_mae': mae,
                'val_mse': mse,
                'val_rmse': rmse}

    ### fitness_linreg ###


    housing_data = pd.read_csv('/media/i-files/data/other-data/housing/housing.csv')
    print(housing_data.shape)
    housing_data = housing_data[housing_data['median_house_value'] < 499_000]
    print(housing_data.shape)

    size = 1000
    X = np.array(housing_data['median_income'][:size])
    y = np.array(housing_data['median_house_value'][:size])

    X = X.reshape(X.shape[0], 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    #regression_model.coef_ = np.array([69708.18])
    #regression_model.intercept_ = 76725.17
    
    benchmark_a1 = regression_model.coef_[0]
    benchmark_a2 = regression_model.intercept_

    print('Coefficient (a1):', benchmark_a1)
    print('Intercept (a2):', benchmark_a2)

    y_line = benchmark_a1 * X_train + benchmark_a2
    y_train_pred = regression_model.predict(X_train)
    y_val_pred = regression_model.predict(X_val)

    mae = mean_absolute_error(y_val, y_val_pred)
    logger.info(f'Mean absolute error: {mae}')

    plt.scatter(X, y)
    #plt.plot(X_train, y_train_pred, color='b')
    plt.plot(X_train, y_line, color='r')
    #plt.plot(X_val, y_val_pred, color='k', linestyle='dashed')
    #plt.show()

    #sys.exit()
    logger.info('X_train.shape = ' + str(X_train.shape) + 'type = ' + str(X_train.dtype))
    logger.info('y_train.shape = ' + str(y_train.shape) + 'type = ' + str(y_train.dtype))
    logger.info('X_val.shape =   ' + str(X_val.shape) + 'type = ' + str(X_val.dtype))
    logger.info('y_val.shape =   ' + str(y_val.shape) + 'type = ' + str(y_val.dtype))

    fitnesses = ['cpu', 'val_mae', 'val_rmse']
    criterion = Criterion(fitnesses[1], 'le', 1.0)

    data = GaData(X_train, X_val, None, y_train, y_val, None)
    data.register_variable('a1', 0)
    data.register_variable('a2', 1)

    fitness = fitness_linreg(data, criterion)
    logger.info('validation mae:  {:.2f}'.format(fitness['val_mae']))
    logger.info('validation mse:  {:.2f}'.format(fitness['val_mse']))
    logger.info('validation rmse: {:.2f}'.format(fitness['val_rmse']))

    pop = Population(p_mutation=0.02, 
                     p_crossover=2, # > 1 means absolute # of crossovers 
                     fitness=fitnesses, 
                     selection_key=criterion.selection_key,
                     n_best_of=1)

    pop.add_var_float('a1', 32, 1, 100_000)
    pop.add_var_float('a2', 32, 1, 100_000) 
    pop.set_fitness_function(fitness_linreg, data)
    pop.create_population(10)

    logger.warning('')
    logger.warning('--- Generation 0 ---')
    pop.pre_compute(criterion)
    pop.show()

    for generation in range(1, 110):
        logger.warning('')
        logger.warning('*** Generation ' + str(generation))
        pop.next_generation(10, criterion)
        pop.show()

    # for

    for key in pop.the_best_of:
        ga = pop.the_best_of[key]
        logger.info(f'{ga.id}')
      
    return    

### ga_demo_linear_regression ###


if __name__ == "__main__":
    logger = create_logger('ga-test.log')

    logger.info('*** GA test')

    #ga_simple_test()
    #ga_simple_array_test()
    #ga_simple_list_test()
    ga_demo_linear_regression()