import sys
import os
import cPickle as pickle
import json
import numpy as np
import getopt
from keras.models import Sequential, Model
from keras.optimizers import SGD
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, Dense, Dropout, Flatten
from keras.applications.vgg16 import VGG16

def usage():
  print "usage: python2 keras_vgg16.py -r <train_file> -n class_num (0/1/2/3/4)"


def levelMapping(target_l1s, target_level):
  with open('catid_to_levelid.json', 'r') as catid_dict_file:
    cat_dict = json.load(catid_dict_file)
  target_level_dict = {}
  for catid in cat_dict:
    val = cat_dict[catid]
    if val[0] == target_l1s:
      target_levelid = val[target_level - 1]
      if target_levelid not in target_level_dict:
        target_level_dict[target_levelid] = len(target_level_dict)
  print len(target_level_dict)
  return target_level_dict


def parseArgs(argv):
  try:
    opts, args = getopt.getopt(argv[1:], 'r:n:',
                               ["train=", "class_num="])
  except getopt.GetoptError:
    usage()
    sys.exit(1)
  train_file = None
  class_num = None
  for o, a in opts:
    if o in ("-r", "--train"):
      train_file = a
    if o in ("-n", "--class_num"):
      class_num = a
  if train_file is None or class_num is None:
    usage()
    sys.exit(1)
  return (train_file, class_num)

def main(argv):
  train_file, class_num = parseArgs(argv)
  target_l1s = np.array([15, 18, 31, 42, 48])
  target = target_l1s[int(class_num)]
  # map actual l3 class label to integer from 0 - l3_size
  l3_dict = levelMapping(target, 3)
  # number of distinct l3 classes
  l3_size = len(l3_dict)
  l3_reverse_mapping = [0 for i in range(l3_size)]
  for key in l3_dict:
    l3_reverse_mapping[l3_dict[key]] = key

  with open(train_file, 'r') as train_data_file:
    train_data = pickle.load(train_data_file)
  print "pickle load data done"

  with open('catid_to_levelid.json', 'r') as dict_file:
    l_dict = json.load(dict_file)

  num_train_original = len(train_data)
  num_train = 0

  for i in range(num_train_original):
    label_tup = l_dict[str(train_data[i][0])]
    if label_tup[0] == target:
      num_train += len(train_data[i][1])
  print "number of training items is", num_train

  train_x = np.empty([num_train, 224, 224, 3])
  train_y_l3 = np.empty([num_train, l3_size])
  image_pos = 0
  for i in range(num_train_original):
    label_tup = l_dict[str(train_data[i][0])]
    l3_label = np.zeros((l3_size,), dtype=float)
    if label_tup[0] == target:
      true_l3_label = label_tup[2]
      l3_label[l3_dict[true_l3_label]] = 1.

      for img in train_data[i][1]:
        train_x[image_pos, :, :, :] = resize(img, (224, 224, 3), mode='edge')
        train_y_l3[image_pos, :] = l3_label
        image_pos += 1

  train_x = np.asarray(train_x)
  train_y_l3 = np.asarray(train_y_l3)
  print "number of training images is", image_pos


  input_layer = Input(shape=(224, 224, 3))
  base_model = VGG16(weights='imagenet', include_top=False)
  x = base_model(input_layer)
  x = Flatten(name='flatten')(x)
  x = Dense(4096, activation='relu', name='fc1')(x)
  x = Dense(4096, activation='relu', name='fc2')(x)
  predictions = Dense(l3_size, activation='softmax')(x)
  model = Model(inputs=input_layer, outputs=predictions)

  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
  # train_x = np.reshape(train_x[:, :, :], (num_train, 224,224, 3))
  model.fit(train_x, train_y_l3[:, :], epochs=20, verbose=1, batch_size = 100, validation_split=0.15)

  model.save('l3_model%d.h5', class_num)


if __name__ == '__main__':
  main(sys.argv)
