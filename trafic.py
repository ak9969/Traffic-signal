import pickle
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
training_file = r'C:\Users\Akshat\Desktop\traficlight\train.p'
validation_file = r'C:\Users\Akshat\Desktop\traficlight\valid.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
import cv2

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
n_validation = len(X_valid)
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
X_train = (X_train - mean_px)/(std_px)
mean_px = X_valid.mean().astype(np.float32)
std_px = X_valid.std().astype(np.float32)
X_valid = (X_valid - mean_px)/(std_px)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train,num_classes=43)
from keras.utils.np_utils import to_categorical
y_valid = to_categorical(y_valid,num_classes=43)
import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
model = Sequential()
model.add(Conv2D(filters = 6, 
                 kernel_size = 5, 
                 strides = 1, 
                 activation = 'relu', 
                 input_shape = (32,32,3)))
model.add(MaxPooling2D(pool_size = 2, strides = 2))
model.add(Conv2D(filters = 16, 
                 kernel_size = 5,
                 strides = 1,	
                 activation = 'relu',
                 input_shape = (14,14,6)))
model.add(MaxPooling2D(pool_size = 2, strides = 2))
model.add(Flatten())
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 84, activation = 'relu'))
model.add(Dense(units = 43, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(X_train ,y_train, validation_data=(X_valid,y_valid), epochs = 42,verbose=1,callbacks=callbacks_list,batch_size = 64)
scores = model.evaluate(X_train, y_train, verbose=0)
