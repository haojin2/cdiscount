import bson
import sys
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, show
import io

class DataReader(object):
  def __init__(self, filename):
    super(DataReader, self).__init__()
    self.data = bson.decode_file_iter(open(filename, 'rb'))

  def getOne(self):
    doc = self.data.next()
    image_lst = []
    for e, pic in enumerate(doc['imgs']):
      image_lst.append(imread(io.BytesIO(pic['picture'])))
    return (int(doc['category_id']), image_lst)

def main(argv):
  rdr = DataReader(argv[1])
  while 1:
    try:
      tup = rdr.getOne()
      for img in tup[1]:
        imshow(img)
        show()
      print 
    except StopIteration:
      break

if __name__ == '__main__':
  main(sys.argv)
