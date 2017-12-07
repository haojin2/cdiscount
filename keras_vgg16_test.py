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
  print "usage: python2 keras_vgg16_test.py -t <test_file> -m <model_file>"


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
  return target_level_dict


def parseArgs(argv):
  try:
    opts, args = getopt.getopt(argv[1:], 't:m:l:',
                               ["test=", "model="])
  except getopt.GetoptError:
    usage()
    sys.exit(1)
  test_file = None
  model_file = None
  for o, a in opts:
    if o in ("-t", "--test"):
      test_file = a
    if o in ("-m", "--model"):
      model_file = a
  if test_file is None or model_file is None:
    usage()
    sys.exit(1)
  return (test_file, model_file)


def loadTestData(test_file, level_dict):
  with open(test_file, 'r') as test_data_file:
    test_data = pickle.load(test_data_file)
  with open('catid_to_levelid.json', 'r') as dict_file:
    cat_dict = json.load(dict_file)

  num_test_item = len(test_data)
  item_image_mapping = np.empty([num_test_item, 2], dtype = int)
  start = 0

  test_x = []
  test_y = []
  
  for i in range(num_test_item):
    label_tup = cat_dict[str(test_data[i][0])]
    label = level_dict[label_tup[2]]

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

  # print "number of test images is {0}".format(str(len(test_x)))
  # print "number of test items is {0}".format(str(num_test_item))
  return test_x, test_y, item_image_mapping



def main(argv):
  test_file, model_file = parseArgs(argv)

  # pick five lv1 categories for training and testing
  target_l1s = set([15, 18, 31, 42, 48])
  # print [elem for elem in target_l1s]
  # sys.exit(1)

  # actual level label -> (0 through n-1)
  level_dict = levelMapping(target_l1s, 3)
  level_size = len(level_dict)

  l3_reverse_mapping = [0 for i in range(level_size)]
  for key in level_dict:
    l3_reverse_mapping[level_dict[key]] = key


  # (0 through n-1) -> actual level label
  level_reverse_mapping = [0 for i in range(level_size)]
  for key in level_dict:
    level_reverse_mapping[level_dict[key]] = key

  # load test data
  test_x, test_y, item_image_mapping = loadTestData(test_file, level_dict)
  print "test data and label ready"

  #############################################################################
  ############################ First Forward Pass #############################
  #############################################################################

  model = load_model(model_file)

  # predict probs and labels
  predicted = model.predict(test_x, verbose = 1)  # (N, 277)
  predicted_label = np.argmax(predicted, axis=1)  # (N, )
  predicted_l3_label_per_item = []

  # prob value of the predicted label
  predicted_prob = np.amax(predicted, axis = 1)

  num_test_item = len(item_image_mapping)
  cl_acc = 0.
  ce_error = 0.

  for i in range(num_test_item):
    true_label = test_y[i]

    # predicted probabilities of true label
    probs = predicted[item_image_mapping[i][0] : item_image_mapping[i][1], true_label]
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
    predicted_l3_label_per_item.append(label)
  print 'Finished first forward prop of test data'
  print 'classification accuracy is ', cl_acc
  print 'cross entropy error is ', ce_error

  #############################################################################
  ############################ Second Forward Pass ############################
  #############################################################################

  cl_acc = 0.
  ce_error = 0.

  l3_reverse_mappings = [] # list of lists
  l3_level_mappings = [] # list of dictionaries
  for i in range(5):
    fname = 'l3_reverse_mapping%d.json' % i
    with open(fname, 'r') as reverse_mapping:
      l3_reverse_mappings.append(json.load(reverse_mapping))
    fname = 'l3_level_mapping%d.json' % i
    with open(fname, 'r') as level_mapping:
      l3_level_mappings.append(json.load(level_mapping))

  # From l3 label use 'l3_to_l1.json' get l1 labels, 
  with open('l3_to_l1.json', 'r') as dict_file:
    l3_to_l1_dict = json.load(dict_file)

  l1_from_l3 = []
  for i in range(len(predicted_l3_label_per_item)):
    general_l3 = l3_reverse_mapping[predicted_l3_label_per_item[i]]
    l1_from_l3.append(l3_to_l1_dict[str(general_l3)])

  # Separate test data in to five groups based on their l3->l1 prediction
  test_xs = [[], [], [], [], []]
  test_ys = [[], [], [], [], []]
  item_image_mappings = [[], [], [], [], []] 

  l1s = [48, 18, 31, 42, 15]
  for i in range(num_test_item): 
    l1 = l1_from_l3[i]
    print l1
    l1_idx = l1s.index(l1)
    start = item_image_mapping[i][0]
    end = item_image_mapping[i][1]

    true_label = l3_reverse_mapping[test_y[i]]
    if str(true_label) in l3_level_mappings[l1_idx]:
      network_label = l3_level_mappings[l1_idx][str(true_label)]
      test_xs[l1_idx].extend(test_x[start : end,:,:,:])
      test_ys[l1_idx].append(network_label) 
      new_end = len(test_xs[l1_idx])
      new_start = new_end - (end - start)
      item_image_mappings[l1_idx].append([new_start, new_end])


  cl_acc = 0.
  # run forward prop for each of 5 level-1 class networks
  for i in range(5):
    if len(test_ys[i]) != 0:
      test_xs[i] = np.asarray(test_xs[i])
      test_ys[i] = np.asarray(test_ys[i])

      model_file = 'l3_model%d.h5' % (i)
      model = load_model(model_file)

      # predict probs and labels
      predicted = model.predict(test_xs[i], verbose = 1, batch_size=1) # (N', out_size)
      predicted_label = np.argmax(predicted, axis=1)  # (N', )

      # prob value of the predicted label
      predicted_prob = np.amax(predicted, axis = 1) # (N', )

      num_items = len(item_image_mappings[i])
      print "runnning %d test items for lv1 class %d" % (num_items,l1s[i])
      for j in range(num_items):
        true_label = test_ys[i][j]

        # predicted probabilities of true label
        start = item_image_mappings[i][j][0]
        end = item_image_mappings[i][j][1]
        probs = predicted[start : end, true_label]

        # predicted labels
        labels = predicted_label[start:end]

        avg_probs = np.average(probs)

        # majority vote
        counts = np.bincount(labels)
        label = np.argmax(counts)

        # if each image vote for a different class,  use the one with highest prob
        if np.max(counts) == 1: 
          label_idx = np.argmax(predicted_prob[start:end])
          label = labels[label_idx]
          # print "tied votes! label idx is {0} and predicted label is {1}".format(str(label_idx), str(label))
        cl_acc += ((label==true_label) * 1./num_test_item)
  del model
  print "classification accuracy is %f" % (cl_acc)


if __name__ == '__main__':
  main(sys.argv)
