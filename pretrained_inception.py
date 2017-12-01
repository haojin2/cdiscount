from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import cPickle as pickle
import json
import keras
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import sys
import urllib
import getopt

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

  #starts here
  train_file, test_file = parseArgs(argv)
  target_l1s = set([5, 13, 18, 19, 23, 28, 31, 32, 46, 48])

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
  
  num_train = len(all_data) 
  num_test = len(test_data)

  train_x = []
  train_y_l3 = []
  for i in range(num_train):
    label_tup = l_dict[str(train_data[i][0])]

    l3_label = np.zeros((l3_size,), dtype=float)
    true_l3_label = label_tup[2]
    l3_label[l3_dict[true_l3_label]] = 1.

    for img in train_data[i][1]:
      train_x.append(resize(img, (224, 224, 3), mode='edge'))
      train_y_l3.append(l3_label)

  train_x = np.asarray(train_x)
  train_y_l3 = np.asarray(train_y_l3)
  #ends here


  # test_x = []
  # test_y_l3 = []

  # for i in range(num_test):
  #   label_tup = l_dict[str(test_data[i][0])]

  #   l3_label = np.zeros((l3_size,), dtype=float)
  #   true_l3_label = label_tup[2]
  #   l3_label[l3_dict[true_l3_label]] = 1.

  #   for img in test_data[i][1]:
  #     test_x.append(resize(img, (224, 224, 3), mode='edge'))
  #     test_y_l3.append(l3_label)

  # test_x = np.asarray(test_x)
  # test_y_l3 = np.asarray(test_y_l3)
  print "all data and label ready"

  base_model = VGG16(weights='imagenet', include_top=False)
  x = base_model.output
  x = Flatten(name='flatten')(x)
  x = Dense(4096, activation='relu', name='fc1')(x)
  x = Dense(4096, activation='relu', name='fc2')(x)
  predictions = Dense(l3_size, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  for layer in base_model.layers:
          layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
  model.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 3)), train_y_l3[:, :], epochs=3, verbose=1, batch_size = 100, validation_split=0.15)
  
  for i, layer in enumerate(model.layers):
    print(i, layer.name)

  for layer in model.layers[:15]:
    layer.trainable = False
  for layer in model.layers[15:19]:
    layer.trainable = True
  for layer in model.layers[19:]:
    layer.trainable = False

  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
  model.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 3)), train_y_l3[:, :], epochs=3, verbose=1, batch_size = 100, validation_split=0.15)


  for layer in model.layers[:11]:
    layer.trainable = False
  for layer in model.layers[11:15]:
    layer.trainable = True
  for layer in model.layers[15:]:
    layer.trainable = False

  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
  model.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 3)), train_y_l3[:, :], epochs=3, verbose=1, batch_size = 100, validation_split=0.15)


  for layer in model.layers[:7]:
    layer.trainable = False
  for layer in model.layers[7:11]:
    layer.trainable = True
  for layer in model.layers[11:]:
    layer.trainable = False

  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
  model.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 3)), train_y_l3[:, :], epochs=3, verbose=1, batch_size = 100, validation_split=0.15)


  for layer in model.layers[:4]:
    layer.trainable = False
  for layer in model.layers[4:7]:
    layer.trainable = True
  for layer in model.layers[7:]:
    layer.trainable = False

  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
  model.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 3)), train_y_l3[:, :], epochs=3, verbose=1, batch_size = 100, validation_split=0.15)

  for layer in model.layers[:1]:
    layer.trainable = False
  for layer in model.layers[1:4]:
    layer.trainable = True
  for layer in model.layers[4:]:
    layer.trainable = False

  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
  model.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 3)), train_y_l3[:, :], epochs=3, verbose=1, batch_size = 100, validation_split=0.15)

  model.save('pretrained_model.h5')
    
if __name__ == '__main__':
    main(sys.argv)


# #Train on 11481 samples, validate on 2027 samples
# Epoch 1/3
# 11481/11481 [==============================] - 42s 4ms/step - loss: 2.1498 - val_loss: 6.3004
# Epoch 2/3
# 11481/11481 [==============================] - 42s 4ms/step - loss: 2.0240 - val_loss: 6.2426
# Epoch 3/3
# 11481/11481 [==============================] - 42s 4ms/step - loss: 1.9346 - val_loss: 6.3349
# Train on 11481 samples, validate on 2027 samples
# Epoch 1/3
# 11481/11481 [==============================] - 49s 4ms/step - loss: 1.8366 - val_loss: 6.4115
# Epoch 2/3
# 11481/11481 [==============================] - 49s 4ms/step - loss: 1.7765 - val_loss: 6.4835
# Epoch 3/3
# 11481/11481 [==============================] - 49s 4ms/step - loss: 1.7391 - val_loss: 6.6366
# Train on 11481 samples, validate on 2027 samples
# Epoch 1/3
# 11481/11481 [==============================] - 57s 5ms/step - loss: 1.6749 - val_loss: 6.6927
# Epoch 2/3
# 11481/11481 [==============================] - 56s 5ms/step - loss: 1.6642 - val_loss: 6.6213
# Epoch 3/3
# 11481/11481 [==============================] - 56s 5ms/step - loss: 1.6541 - val_loss: 6.6310
# Train on 11481 samples, validate on 2027 samples
# Epoch 1/3
# 11481/11481 [==============================] - 61s 5ms/step - loss: 1.6365 - val_loss: 6.6645
# Epoch 2/3
# 11481/11481 [==============================] - 61s 5ms/step - loss: 1.6364 - val_loss: 6.6841
# Epoch 3/3
# 11481/11481 [==============================] - 61s 5ms/step - loss: 1.6329 - val_loss: 6.6746
# Train on 11481 samples, validate on 2027 samples
# Epoch 1/3
# 11481/11481 [==============================] - 71s 6ms/step - loss: 1.6270 - val_loss: 6.6667
# Epoch 2/3
# 11481/11481 [==============================] - 71s 6ms/step - loss: 1.6268 - val_loss: 6.6853
# Epoch 3/3
# 11481/11481 [==============================] - 71s 6ms/step - loss: 1.6259 - val_loss: 6.6300
