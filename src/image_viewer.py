# based on https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import sys
import h5py
import time
import random
import numpy as np
import matplotlib.pyplot as plt
# import cv2 needs python 3.5-3.7, incompatible with tensorflow
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

from ga import Data, Population

# Code initialisatie: logging
import logging
import importlib
importlib.reload(logging)

# create logger
logger = logging.getLogger('ga')

logger.setLevel(10)

# create file handler which logs even debug messages
fh = logging.FileHandler('ga.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(logging.Formatter('%(message)s'))


def read_h5 (h5_path):
    with h5py.File(h5_path, "r") as hdf5_file:
        n = hdf5_file["images"].shape[0]
        print ('hdf5 file shape', hdf5_file["images"].shape)

        images = np.array (hdf5_file['images'][0:n])
        labels = np.array(hdf5_file['labels'])
        names = np.array(hdf5_file['names'])
        labels = labels.reshape(len(labels), 1)
        n_categories = labels.max()
        if n_categories > 1:
            labels = np.asarray(OneHotEncoder(sparse=False).fit_transform (labels).astype(np.int32))

        return images, labels, names
    # with
### read_h5 ###

if __name__ == "__main__":
    print('\n\n=========================================')
    data_file: str = "/media/i/home/data/pid/pid-3cat-256x256x3.h5"

    # read the data
    X, y, _ = read_h5(data_file)
   
    # X = np.array([cv2.resize(x, (128, 128)) for x in X[:]])
    
    #X = np.array([cv2.resize(x, (128, 128)) for x in X[:]])
    #img = Image.frombytes('RGB', (256, 256), X[0]).resize((128, 128), 
    #                                                      resample=Image.HAMMING,
    #                                                      reducing_gap=5)
    
    img = Image.fromarray(X[0]).resize((128, 128), 
                                       resample=Image.HAMMING,
                                       reducing_gap=5)
    
    Xx = np.array([np.asarray(Image.fromarray(x)
                   .resize((128, 128), 
                            resample=Image.HAMMING,
                            reducing_gap=5))
                  for x in X[:]])
    
    print('==>', Xx.shape)

    plt.imshow(Xx[random.randint(0, 2500)])
    plt.show()