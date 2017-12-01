from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
import cPickle as pickle
import json
import keras
import numpy as np
from skimage.transform import resize
import sys
import urllib
import getopt

def usage():
  print "usage: python2 test_model.py -t <test_file> -m <saved_model>"

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
    opts, args = getopt.getopt(argv[1:], 't:m:',
                               ["test=","model="])
  except getopt.GetoptError:
    usage()
    sys.exit(1)
  test_file = None
  saved_model = None
  for o, a in opts:
    if o in ("-t", "--test"):
      test_file = a
    if o in ("-m", "--model"):
      saved_model = a
  if saved_model is None or test_file is None :
    usage()
    sys.exit(1)
  return (test_file, saved_model)


def main(argv):

  #starts here
  test_file, saved_model= parseArgs(argv)
  target_l1s = set([15, 18, 31, 42, 48])

  l3_dict = levelMapping(target_l1s, 3)
  l3_size = len(l3_dict)
  l3_reverse_mapping = [0 for i in range(l3_size)]
  for key in l3_dict:
    l3_reverse_mapping[l3_dict[key]] = key

  with open(test_file, 'r') as test_data_file:
    test_data = pickle.load(test_data_file)
  print "load data done"

  with open('catid_to_levelid.json', 'r') as dict_file:
    l_dict = json.load(dict_file)
  
  num_test = len(test_data)
  test_x = []
  test_y_l3 = []
  item_image_mapping = np.empty([num_test, 2], dtype = int)
  start = 0
  
  for i in range(num_test):
    num_img = len(test_data[i])
    item_image_mapping[i, 0] = start
    item_image_mapping[i, 1] = start + num_img
    start = start + num_img

    label_tup = l_dict[str(test_data[i][0])]
    l3_label = np.zeros((l3_size,), dtype=float)
    true_l3_label = label_tup[2]
    l3_label[l3_dict[true_l3_label]] = 1.
    test_y_l3.append(l3_label)
    for img in test_data[i][1]:
      test_x.append(resize(img, (224, 224, 3), mode='edge'))
      
  test_x = np.asarray(test_x)
  test_y_l3 = np.asarray(test_y_l3)

  print "all data and label ready"
  model = keras.models.load_model(saved_model)

  predicted_ = model.predict(test_x, verbose = 1)
  predicted_prob = np.amax(predicted_, axis = 1)
  predicted_label = np.argmax(predicted_, axis=1)

  cl_acc = 0.
  ce_error = 0.

  for i in range(num_test):
    res = predicted_label[item_image_mapping[i][0]:item_image_mapping[i][1]]
    true_label = test_y_l3[i]

    probs = predicted_[item_image_mapping[i][0]:item_image_mapping[i][1], true_label]
    avg_probs = np.average(probs)
    ce_error += -np.log(avg_probs) /num_test

    counts = np.bincount(res)
    label = np.argmax(counts)
    if np.max(counts) == 1:
      label = np.argmax(predicted_prob[item_image_mapping[i][0]:item_image_mapping[i][1]]
    cl_acc[i] += ((label==true_label) * 1./num_test)

  print 'classification accuracy is ', cl_acc
  print 'cross entropy error is ', ce_error

    
if __name__ == '__main__':
    main(sys.argv)

