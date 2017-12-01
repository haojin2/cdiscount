import sys
import os
import cPickle as pickle
import json
import numpy as np
import getopt
from keras.models import Sequential
from keras.optimizers import SGD
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Dropout, Flatten
from keras.applications.vgg16 import VGG16

def usage():
  print "usage: python2 keras_vgg16.py -r <train_file> -t <test_file>"


def levelMapping(target_l1s, target_level):
  with open('catid_to_levelid.json', 'r') as catid_dict_file:
    cat_dict = json.load(catid_dict_file)
  target_level_dict = {}
  for catid in cat_dict:
    val = cat_dict[catid]
    if val[0] in target_l1s:
      target_levelid = val[target_level - 1]
      if target_levelid not in target_level_dict:
        target_level_dict[target_levelid] = len(target_level_dict)
  print len(target_level_dict)
  return target_level_dict


def parseArgs(argv):
  try:
    opts, args = getopt.getopt(argv[1:], 'r:t:',
                               ["train=", "test="])
  except getopt.GetoptError:
    usage()
    sys.exit(1)
  train_file = None
  test_file = None
  for o, a in opts:
    if o in ("-r", "--train"):
      train_file = a
    if o in ("-t", "--test"):
      test_file = a
  if train_file is None or test_file is None:
    usage()
    sys.exit(1)
  return (train_file, test_file)

def main(argv):
  train_file, test_file = parseArgs(argv)
  target_l1s = set([15, 18, 31, 42, 48])

  l1_dict = levelMapping(target_l1s, 1)
  l1_size = len(l1_dict)
  l1_reverse_mapping = [0 for i in range(l1_size)]
  for key in l1_dict:
    l1_reverse_mapping[l1_dict[key]] = key

  l2_dict = levelMapping(target_l1s, 2)
  l2_size = len(l2_dict)
  l2_reverse_mapping = [0 for i in range(l2_size)]
  for key in l2_dict:
    l2_reverse_mapping[l2_dict[key]] = key

  l3_dict = levelMapping(target_l1s, 3)
  l3_size = len(l3_dict)
  l3_reverse_mapping = [0 for i in range(l3_size)]
  for key in l3_dict:
    l3_reverse_mapping[l3_dict[key]] = key

  with open(train_file, 'r') as train_data_file:
    train_data = pickle.load(train_data_file)
  with open(test_file, 'r') as test_data_file:
    test_data = pickle.load(test_data_file)
  print "load data done"

  with open('catid_to_levelid.json', 'r') as dict_file:
    l_dict = json.load(dict_file)

  num_train = len(train_data)
  num_test = len(test_data)

  train_x = []
  train_y_l1 = []
  train_y_l2 = []
  train_y_l3 = []
  for i in range(num_train):
    label_tup = l_dict[str(train_data[i][0])]
    l1_label = np.zeros((l1_size,), dtype=float)
    true_l1_label = label_tup[0]
    l1_label[l1_dict[true_l1_label]] = 1.

    l2_label = np.zeros((l2_size,), dtype=float)
    true_l2_label = label_tup[1]
    l2_label[l2_dict[true_l2_label]] = 1.

    l3_label = np.zeros((l3_size,), dtype=float)
    true_l3_label = label_tup[2]
    l3_label[l3_dict[true_l3_label]] = 1.

    for img in train_data[i][1]:
      train_x.append(resize(img, (224, 224, 3), mode='edge'))
      train_y_l1.append(l1_label)
      train_y_l2.append(l2_label)
      train_y_l3.append(l3_label)

  print "train data and label ready"
  image_shape = train_x[0].shape

  test_x = []
  test_y_l1 = []
  test_y_l2 = []
  test_y_l3 = []
  for i in range(num_test):
    label_tup = l_dict[str(test_data[i][0])]
    l1_label = np.zeros((l1_size,), dtype=float)
    true_l1_label = label_tup[0]
    l1_label[l1_dict[true_l1_label]] = 1.

    l2_label = np.zeros((l2_size,), dtype=float)
    true_l2_label = label_tup[1]
    l2_label[l2_dict[true_l2_label]] = 1.

    l3_label = np.zeros((l3_size,), dtype=float)
    true_l3_label = label_tup[2]
    l3_label[l3_dict[true_l3_label]] = 1.

    for img in test_data[i][1]:
      test_x.append(resize(img, (224, 224, 3), mode='edge'))
      test_y_l1.append(l1_label)
      test_y_l2.append(l2_label)
      test_y_l3.append(l3_label)

  num_train = len(train_x)
  train_x = np.asarray(train_x)
  train_y_l1 = np.asarray(train_y_l1)
  train_y_l2 = np.asarray(train_y_l2)
  train_y_l3 = np.asarray(train_y_l3)

  num_test = len(test_x)
  test_x = np.asarray(test_x)
  test_y_l1 = np.asarray(test_y_l1)
  test_y_l2 = np.asarray(test_y_l2)
  test_y_l3 = np.asarray(test_y_l3)
  print "all data and label ready"
  sys.exit(0)

  model1 = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=l1_size)
  model2 = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=l2_size)
  model3 = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=l3_size)

  sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.5, nesterov=True)

  model1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  model1.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 3)), train_y_l1[:, :], validation_split=0.15, epochs=200, verbose=2, batch_size = 100)

  model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  model2.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 3)), train_y_l2[:, :], validation_split=0.15, epochs=200, verbose=2, batch_size = 100)

  model3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  model3.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 3)), train_y_l3[:, :], validation_split=0.15, epochs=200, verbose=2, batch_size = 100)

  model1.save('l1_model.h5')
  model2.save('l2_model.h5')
  model3.save('l3_model.h5')


if __name__ == '__main__':
  main(sys.argv)
