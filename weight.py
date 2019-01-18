import keras 
import pickle
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import cv2
from PIL import Image
#import trafic.py
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
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
model.load_weights(r"C:\Users\Akshat\Desktop\traficlight\weights.best.hdf5")
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#print(model.predict(img))
#print(np.argmax(model.predict(img)))
i = 0
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(33)
    if k==27:  #Press ESC to exit
        break
    elif k == 32: #Press space bar to enter
        frame  = cv2.resize(frame,(32,32))
        frame = np.expand_dims(frame,axis=0)
        print(model.predict(frame))
        print(np.argmax(model.predict(frame)))
    i=i+1    
