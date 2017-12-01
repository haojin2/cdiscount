import pickle
import json
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import numpy as np
import sys
import urllib
#https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
# folder_path = "http://paaaandayuan.s3.amazonaws.com/707/"
#folder_path = "/Volumes/APPLE-RED/707/"
def main(argv):
  with open(argv[1], 'r') as data_file:
    all_data = pickle.load(data_file)
  with open('catid_to_levelid.json', 'r') as dict_file:
    l_dict = json.load(dict_file)
  print len(all_data)
  num_train = len(all_data) 
  x = []
  y = []
  for i in range(num_train):
    x.append(resize(rgb2gray(all_data[i][1][0]), (224,224), mode='reflect'))
    label = np.zeros((49,), dtype=float)
    label[l_dict[str(all_data[i][0])][0]] = 1.
    y.append(label)
  image_shape = x[0].shape
  x = np.asarray(x)
  y = np.asarray(y)
 
  #np.savetxt("x.csv", x, delimiter=",")
  #np.savetxt("y.csv", y, delimiter=",")
  model = Sequential()
  model.add(Conv2D(64,(3,3),activation ='relu', input_shape=(224,224, 1)))
  model.add(Conv2D(64,(3,3),activation ='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Conv2D(128,(3,3),activation ='relu'))
  model.add(Conv2D(128,(3,3),activation ='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Conv2D(256,(3,3),activation ='relu'))
  model.add(Conv2D(256,(3,3),activation ='relu'))
  model.add(Conv2D(256,(3,3),activation ='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Conv2D(512,(3,3),activation ='relu'))
  model.add(Conv2D(512,(3,3),activation ='relu'))
  model.add(Conv2D(512,(3,3),activation ='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dense(4096, activation='relu'))
  model.add(Dense(49, activation='softmax'))

  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5)
  model.compile(optimizer=sgd, loss='categorical_crossentropy')
  model.fit(np.reshape(x[:, :, :], (num_train, 224,224, 1)), y[:, :], epochs=200, verbose=1, batch_size = 20)

if __name__ == '__main__':
  main(sys.argv)