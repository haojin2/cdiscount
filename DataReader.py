import bson
import sys
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, show
import io

# data = bson.decode_file_iter(open('/media/michael/Seagate Backup Plus Drive/data/train.bson', 'rb'))
data = bson.decode_file_iter(open('train_example.bson', 'rb'))
cnt = 0

for c, d in enumerate(data):
  # print d['_id'], d['category_id']
  cnt += 1
  for e, pic in enumerate(d['imgs']):
    picture = imread(io.BytesIO(pic['picture']))
    imshow(picture)
    show()
    # do something with the picture, etc
  print cnt

print cnt

# def main(argv):


# if __name__ == '__main__':
#   main(sys.argv)
