import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score, f1_score


def read_data():
    file_name = '/media/i-files/data/mnist/train.csv'
    print('Reading data from:', file_name)
    raw = pd.read_csv(file_name, sep = ',', header=None, index_col=None).to_numpy()
    X = raw[:, 1:]
    y = raw[:, :1]
    print('getdata_pid\nRaw.shape = ' + str(raw.shape) + 'type = ' + str(raw.dtype))
    print('X.shape = ' + str(X.shape) + 'type = ' + str(X.dtype))
    print('y.shape = ' + str(y.shape) + 'type = ' + str(y.dtype))
    
    return X, y


# Definition of global constants
n_features = 28 * 28
layer_sizes = [10, 20]

# load a simple data set
X, y = read_data()

# normalize and one hot encode the data
X = X / 255.0
y = OneHotEncoder (sparse=False).fit_transform (y).astype (np.int32)

# split in training and test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33)

# Rescale the predictors for better performance
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_val = sc_X.transform(X_val)

cpu = time.time()

input_nodes = X_train.shape[1]
output_nodes = y_train.shape[1]

# Create layers
lagen = []
print('Layers:', layer_sizes)
for units in layer_sizes:
    #model.add(
    lagen.append(
                Dense(units, 
                    kernel_initializer='glorot_normal', 
                    input_dim=input_nodes, 
                    activation='relu')
                )

    print(f'Added layer to model with size {units}')
    
# Add final Dense layer
lagen.append(Dense(output_nodes, activation='softmax'))#, kernel_initializer='glorot_normal')
print(f'Added dense layer of size {output_nodes}')

# Create model
cpu = time.time()
model = Sequential(lagen)
cpu = time.time() - cpu
print(f'Model created in {cpu:.2f} seconds')

adam = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print('Compiled model')

model.summary()

cpu = time.time()
print('Starting to fit model')
hist = model.fit(X_train, y_train,   
                validation_data=(X_val, y_val), 
                epochs=10,
                batch_size=256,
                verbose=1)

cpu = time.time() - cpu
print(f'CPU is {cpu:.2f}')

y_pred = model.predict(X_val)

val_label = np.argmax(y_val, axis=1) 
pred_label = np.argmax(y_pred, axis=1)

val_acc = accuracy_score(val_label, pred_label, normalize=True)
f1 = f1_score(val_label, pred_label, average='weighted')
acc_cpu = val_acc / cpu
print('validation accuracy: {:.2f}'.format(val_acc))
print('validation F1:       {:.2f}'.format(f1))
print('val accuracy / cpu:  {:.2f}'.format(acc_cpu))