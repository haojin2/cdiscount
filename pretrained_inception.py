from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
import pickle
import json
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import sys
import urllib



folder_path = "http://paaaandayuan.s3.amazonaws.com/707/"
#folder_path = "/Volumes/APPLE-RED/707/"
def main(argv):
	data_file = urllib.URLopener().open(folder_path + argv[1])
	# with open(folder_path + argv[1], 'r') as data_file:
	all_data = pickle.load(data_file)
	with open('catid_to_levelid.json', 'r') as dict_file:
		l_dict = json.load(dict_file)
	print len(all_data)
	num_train = len(all_data) 
	train_x = []
	train_y = []
	for i in range(num_train):
		train_x.append(resize(rgb2gray(all_data[i][1][0]), (224,224), mode='reflect'))
		label = np.zeros((49,), dtype=float)
		label[l_dict[str(all_data[i][0])][0]] = 1.
		train_y.append(label)
	image_shape = x[0].shape
	print image_shape
	train_x = np.asarray(x)
	train_y = np.asarray(y)

	# create the base pre-trained model
	base_model = InceptionV3(weights='imagenet', include_top=False)

	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	# and a logistic layer -- let's say we have 200 classes
	predictions = Dense(49, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional InceptionV3 layers
	for layer in base_model.layers:
			layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	# train the model on the new data for a few epochs
	# model.fit_generator(...)
	model.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 1)), train_y[:, :], epochs=5, verbose=1, batch_size = 20)

	# at this point, the top layers are well trained and we can start fine-tuning
	# convolutional layers from inception V3. We will freeze the bottom N layers
	# and train the remaining top layers.

	# we chose to train the top 2 inception blocks, i.e. we will freeze
	# the first 249 layers and unfreeze the rest:
	for layer in model.layers[:249]:
		 layer.trainable = False
	for layer in model.layers[249:]:
		 layer.trainable = True

	# we need to recompile the model for these modifications to take effect
	# we use SGD with a low learning rate
	model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

	# we train our model again (this time fine-tuning the top 2 inception blocks
	# alongside the top Dense layers
	model.fit(np.reshape(train_x[:, :, :], (num_train, 224,224, 1)), train_y[:, :], epochs=10, verbose=1, batch_size = 20)

if __name__ == '__main__':
	main(sys.argv)