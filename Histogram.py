import sys
import os
import json
import pickle
from DataReader import DataReader

def main(argv):
  try:
    rdr = DataReader(argv[1])
    level = int(argv[2])
  except:
    print "usage: python Extractor.py <bson_file> <level>"
  print level
  with open('catid_to_levelid.json', 'r') as ldict_file:
    cat_dict = json.load(ldict_file)
  with open('l%d_dict.json' % level, 'r') as ldict_file:
    l_dict = json.load(ldict_file)
  category_cnts = [0 for i in xrange(len(l_dict))]
  category_img_cnts = [0 for i in xrange(len(l_dict))]
  cnt = 1
  while 1:
    try:
      tup = rdr.getOne()
      if cnt % 10000 == 0:
        print cnt
      cnt += 1
      catid = tup[0]
      cat_id = cat_dict[str(catid)][level-1]
      category_img_cnts[cat_id] += len(tup[1])
    except StopIteration:
      print "none"
      break
  with open("l%d_histo.json" % level, 'r') as histo_file:
    json.dump(histo_file, category_img_cnts)


if __name__ == '__main__':
  main(sys.argv)