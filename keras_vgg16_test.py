import sys
import os
import cPickle as pickle
import json
import numpy as np
import getopt
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Dropout, Flatten
from keras.applications.vgg16 import VGG16

def usage():
  print "usage: python2 keras_vgg16_test.py -t <test_file> -m <model_file> -l <level>"


# Map labels of target level to integers 0 through n-1 where n is the
# total number of distinct labels in target level whose l1 ID is in target_l1s
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
    opts, args = getopt.getopt(argv[1:], 't:m:l:',
                               ["test=", "model=", "level="])
  except getopt.GetoptError:
    usage()
    sys.exit(1)
  test_file = None
  model_file = None
  level = None
  for o, a in opts:
    if o in ("-t", "--test"):
      test_file = a
    if o in ("-m", "--model"):
      model_file = a
    if o in ("-l", "--level"):
      level = int(a)
  if test_file is None or model_file is None or level is None:
    usage()
    sys.exit(1)
  return (test_file, model_file, level)


def loadTestData(test_file, level, level_dict):
  with open(test_file, 'r') as test_data_file:
    test_data = pickle.load(test_data_file)
  with open('catid_to_levelid.json', 'r') as dict_file:
    cat_dict = json.load(dict_file)

  num_test_item = len(test_data)
  # num_test_item = 10
  item_image_mapping = np.empty([num_test_item, 2], dtype = int)
  start = 0

  test_x = []
  test_y = []
  
  for i in range(num_test_item):
    label_tup = cat_dict[str(test_data[i][0])]
    label = level_dict[label_tup[level - 1]]

    # mark the start(inclusive) and end(non-inclusive) of test 
    # images that belong to the same item
    num_img = len(test_data[i][1])
    item_image_mapping[i, 0] = start
    item_image_mapping[i, 1] = start + num_img
    start = start + num_img

    # append test input and test label
    test_y.append(label)
    for img in test_data[i][1]:
      test_x.append(resize(img, (224, 224, 3), mode='edge'))
    
  test_x = np.asarray(test_x)
  test_y = np.asarray(test_y)

  print "number of test images is {0}".format(str(len(test_x)))
  print "number of test items is {0}".format(str(num_test_item))
  return test_x, test_y, item_image_mapping



def main(argv):
  test_file, model_file, level = parseArgs(argv)

  # pick five lv1 categories for training and testing
  target_l1s = set([15, 18, 31, 42, 48])

  # actual level label -> 0 through n-1
  level_dict = levelMapping(target_l1s, level)
  level_size = len(level_dict)
  # 0 through n-1 -> actual level label
  level_reverse_mapping = [0 for i in range(level_size)]
  for key in level_dict:
    level_reverse_mapping[level_dict[key]] = key

  test_x, test_y, item_image_mapping = loadTestData(test_file, level, level_dict)
  print "test data and label ready"

  model = load_model(model_file)

  # predict test label
  predicted = model.predict(test_x, verbose = 1)
  predicted_label = np.argmax(predicted, axis=1)
  # prob value of the predicted label
  predicted_prob = np.amax(predicted, axis = 1)

  num_test_item = len(item_image_mapping)

  cl_acc = 0.
  ce_error = 0.

  for i in range(num_test_item):
    true_label = test_y[i]

    # predicted probabilities of true label
    start = item_image_mapping[i][0]
    end = item_image_mapping[i][1]
    print start, end
    probs = predicted[start:end, true_label]
    # predicted labels
    labels = predicted_label[item_image_mapping[i][0]:item_image_mapping[i][1]]

    avg_probs = np.average(probs)
    ce_error += -np.log(avg_probs) /num_test_item

    # majority vote
    counts = np.bincount(labels)
    label = np.argmax(counts)

    # if each image vote for a different class,  use the one with highest prob
    if np.max(counts) == 1: 
      label_idx = np.argmax(predicted_prob[item_image_mapping[i][0]:item_image_mapping[i][1]])
      label = labels[label_idx]
      # print "tied votes! label idx is {0} and predicted label is {1}".format(str(label_idx), str(label))
    cl_acc += ((label==true_label) * 1./num_test_item)

  print 'classification accuracy is ', cl_acc
  print 'cross entropy error is ', ce_error

if __name__ == '__main__':
  main(sys.argv)
