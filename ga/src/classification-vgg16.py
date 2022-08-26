# based on https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
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
from PIL import Image

import tensorflow as tf
from tensorflow.python.client import device_lib

import keras
import keras.backend as K
from keras import mixed_precision
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.losses import categorical_crossentropy
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

from ga import Population
from ga import GaData as Data

# Code initialisatie: logging
import logging
import importlib
importlib.reload(logging)

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

seed = 42
random.seed(seed)

#Allow GPU Growth
'''
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tfconfig.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=tfconfig)
set_session(sess)  # set this TensorFlow session as the default session for Keras

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
'''
#K.set_floatx('float16')
#policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
#tf.keras.mixed_precision.experimental.set_policy(policy)
#K.set_epsilon(1e-4) #default is 1e-7

def plot(hist):
    plt.plot(hist.history["acc"])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.show()

    img = image.load_img("image.jpeg",target_size=(224,224))
    img = np.asarray(img)
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)

    saved_model = load_model("vgg16_1.h5")
    output = saved_model.predict(img)
    if output[0][0] > output[0][1]:
        print("cat")
    else:
        print('dog')
        
    return

### plot ###
    
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

def classify_vgg16(data: Data):
    cpu = time.time()
    # fetch the ML data
    X_train = data.X_train
    X_val = data.X_val
    y_train = data.y_train
    y_val = data.y_val
    
    # get the NN configaration data
    verbose = data.data_dict['verbose']
    epochs = data.data_dict['n_epochs']
    batch_size = data.data_dict['batch_size']
    n_layers = data.data_dict['n_layers']
    layer_sizes = []
    filter_sizes = []
    kernel_sizes = []
    
    for i in range(n_layers):
        #layer_sizes.append(data.data_dict[f'layer_size_{i:d}'])
        filter_sizes.append(data.data_dict[f'filter_size_{i:d}'])
        #kernel_sizes.append(data.data_dict[f'kernel_size_{i:d}'])
    
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    model = Sequential()

    model.add(Conv2D(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), 
                     filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    
    f0 = 2 ** filter_sizes[0]
    model.add(Conv2D(filters=f0, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    f1 = 2 ** filter_sizes[1]
    model.add(Conv2D(filters=f1, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=f1, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    f2 = 2 ** filter_sizes[2]
    model.add(Conv2D(filters=f2, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=f2, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=f2, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    f3 = 2 ** filter_sizes[3]
    model.add(Conv2D(filters=f3, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=f3, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=f3, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    f4 = 2 ** filter_sizes[4]
    model.add(Conv2D(filters=f4, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=f4, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=f4, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=y.shape[1], activation="softmax"))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, 
                loss=categorical_crossentropy, 
                metrics=['accuracy'])

    if verbose == 1:
        model.summary()

    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', 
                                verbose=verbose, save_best_only=True, 
                                save_weights_only=False, mode='auto',) 
                                #save_freq='epoch')

    early = EarlyStopping(monitor='val_accuracy', min_delta=0.00001, 
                        patience=5, verbose=verbose, mode='auto')

    try:
        hist = model.fit(train_generator,   
                        validation_data=(X_val, y_val), #val_generator, 
                        validation_steps=10,
                        epochs=epochs,
                        steps_per_epoch=100,
                        callbacks=[checkpoint, early],
                        verbose=verbose)

        best_model = load_model('vgg16_1.h5')
        y_pred = best_model.predict(X_val)

        val_label = np.argmax(y_val, axis=1) # act_label = 1 (index)
        pred_label = np.argmax(y_pred, axis=1) # pred_label = 1 (index)

        val_acc = accuracy_score(val_label, pred_label, normalize=True)
        f1 = f1_score(val_label, pred_label, average='weighted')
    
    
    except:
        val_acc = -1
        f1 = -1
    
    # try..except
            
    return [val_acc, f1]

### classify_vgg16 ###

if __name__ == "__main__":
    print('\n\n=========================================')
    # print devices on which tensorflow can be computed
    print("Number of GPU's available:", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())
    
    data_file: str = "/media/i/home/data/pid/pid-3cat-256x256x3.h5"
    train_fraction: float = 0.66
    batch_size: int = 8
    n_epochs: int = 10

    # read the data
    X, y, _ = read_h5(data_file)
        
    # resize images for speed and memory reduction
    #X = np.array([cv2.resize(x, (128, 128)) for x in X[:]])
    X = np.array([np.asarray(Image.fromarray(x)
                   .resize((128, 128), 
                            resample=Image.HAMMING,
                            reducing_gap=5))
                  for x in X[:]])  
      
    # split data in training and validation set
    train_size: int = int(train_fraction * X.shape[0])
    val_size: int = X.shape[0] - train_size
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                        train_size=train_size, test_size=val_size,
                        shuffle=True, stratify=y,
                        random_state=seed)

    print('\nData prepared and split\nX_train.shape =', X_train.shape, 'type =', X_train.dtype)
    print('y_train.shape =', y_train.shape, 'type =', y_train.dtype)
    print('X_val.shape =', X_val.shape, 'type = ', X_val.dtype)
    print('y_val.shape =', y_val.shape, 'type =', y_val.dtype)
        
    # set parameter for log regression
    n_layers: int = 5
    layer_size: list = [3, 3, 3, 3, 3]
    filter_size: list = [6, 7, 8, 9, 9]
    kernel_size: list = [3, 3, 3, 3, 3]
    densities: int = 2
    
    fitnesses = ['val_acc', 'val_f1']
    pop = Population(p_mutation=0.25, p_crossover=2, 
                     fitness=fitnesses, selection_key=fitnesses[0])
    
    data = Data(X_train, X_val, None, y_train, y_val, None)
    data.register_variable('verbose', 1)
    data.register_variable('n_epochs', 1)
    data.register_variable('batch_size', batch_size)
    data.register_variable('n_layers', n_layers)
    for i in range(n_layers):
        #var = f'layer_size_{i:d}'
        #data.register_variable(var, layer_size[i])
        #pop.add_var(var, 4, 'I', 1, 5)  
          
        var = f'filter_size_{i:d}'
        data.register_variable(var, filter_size[i])
        pop.add_var(var, 4, 'I', 4, 9)  
        
        #var = f'kernel_size_{i:d}'
        #data.register_variable(var, kernel_size[i])
        #pop.add_var(var, 4, 'I', 1, 5)  

    fitness = classify_vgg16(data)
    print('validation accuracy: {:.2f}'.format(fitness[0]))
    print('validation F1:       {:.2f}'.format(fitness[1]))

    #sys.exit()
    
    data.data_dict['verbose'] = 0
    data.data_dict['n_epochs'] = n_epochs
    
    pop.set_fitness_function(classify_vgg16, data)
    pop.create_population(5)
    
    logger.info('--- Generation 0 ---')

    # pre-compute the fitnesses for the first time because that has
    # not yet been computed and we still want to see it in show()
    pop.get_fitnesses(pop.population, fitnesses[0])
    pop.show()
    
    cpu = time.time()
    for generation in range(1, 10):
        logger.warning('')
        pop.next_generation(5)
        logger.warning('*** Generation {:d} in {:.2f} seconds ***'.
                       format(generation, time.time() - cpu))
        pop.show()
      
    logger.info('')  
    pop.show()


