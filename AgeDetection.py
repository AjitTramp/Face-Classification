data_dir='/home/ajit/Desktop/AgeDetection'

% pylab inline
import os
import random

import pandas as pd
from scipy.misc import imread

root_dir = os.path.abspath('.')

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

import cv2 
temp = []
for img_name in train.ID:
    img_path = os.path.join(data_dir, 'Train', img_name)
    img = imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') # this will help us in later stage
    temp.append(img)

train_x = np.stack(temp)

temp = []
for img_name in test.ID:
    img_path = os.path.join(data_dir, 'Test', img_name)
    img = imread(img_path)
    img = cv2.resize(img, (32, 32))
    temp.append(img.astype('float32'))

test_x = np.stack(temp)

train_x = train_x / 255.
test_x = test_x / 255.

# First Submission 
train.Class.value_counts(normalize=True)
test['Class'] = 'MIDDLE'
test.to_csv(data_dir+"/"+'sub01.csv', index=False)

#Using CNN
import keras
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 256
num_classes = 3
epochs = 80

#input image dimensions
img_rows, img_cols = 32,32

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_initializer='he_normal',input_shape=(32,32,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()

model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1, validation_split=0.2)
pred = model.predict_classes(test_x)
pred = lb.inverse_transform(pred)

test['Class'] =pred

test.to_csv(data_dir+"/"+'sub02.csv', index=False)
