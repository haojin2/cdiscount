import pickle
import json
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from skimage.color import rgb2gray
import numpy as np
import sys
#https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
folder_path = ""

def main(argv):
  with open(folder_path + argv[1], 'r') as data_file:
    all_data = pickle.load(data_file)
  with open('catid_to_levelid.json', 'r') as dict_file:
    l_dict = json.load(dict_file)
  print len(all_data)
  num_train = len(all_data) 
  x = []
  y = []
  for i in range(num_train):
    x.append(rgb2gray(all_data[i][1][0]))
    label = np.zeros((49,), dtype=float)
    label[l_dict[str(all_data[i][0])][0]] = 1.
    y.append(label)
  image_shape = x[0].shape
  print image_shape
  x = np.asarray(x)
  y = np.asarray(y)
  print x.shape
  print y.shape
  model = Sequential()
  model.add(Convolution2D(32,3,3,activation ='relu', input_shape=(180, 180, 1)))
  model.add(MaxPooling2D((2,2), strides=(2,2)))
  # model.add(Convolution2D(128,3,3, activation ='relu'))
  # model.add(MaxPooling2D((2,2), strides=(2,2)))
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(49, activation='softmax'))

  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5)
  model.compile(optimizer=sgd, loss='categorical_hinge')
  model.fit(np.reshape(x[:, :, :], (num_train, 180, 180, 1)), y[:, :], epochs=200, verbose=1)

if __name__ == '__main__':
  main(sys.argv)