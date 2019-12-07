# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate

def shallownet_sequential(width, height, depth, classes):
	# initialize the model along with the input shape to be
	# "channels last" ordering
	model = Sequential()
	inputShape = (height, width, depth)

	# define the first (and only) CONV => RELU layer
	model.add(Conv2D(32, (3, 3), padding="same",
		input_shape=inputShape))
	model.add(Activation("relu"))

	# softmax classifier
	model.add(Flatten())
	model.add(Dense(classes))
	model.add(Activation("softmax"))

	# return the constructed network architecture
	return model

def minigooglenet_functional(width, height, depth, classes):
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
		# define a CONV => BN => RELU pattern
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)

		# return the block
		return x

	def inception_module(x, numK1x1, numK3x3, chanDim):
		# define two CONV modules, then concatenate across the
		# channel dimension
		conv_1x1 = conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
		conv_3x3 = conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

		# return the block
		return x

	def downsample_module(x, K, chanDim):
		# define the CONV module and POOL, then concatenate
		# across the channel dimensions
		conv_3x3 = conv_module(x, K, 3, 3, (2, 2), chanDim,
			padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		x = concatenate([conv_3x3, pool], axis=chanDim)

		# return the block
		return x

	# initialize the input shape to be "channels last" and the
	# channels dimension itself
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input and first CONV module
	inputs = Input(shape=inputShape)
	x = conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

	# two Inception modules followed by a downsample module
	x = inception_module(x, 32, 32, chanDim)
	x = inception_module(x, 32, 48, chanDim)
	x = downsample_module(x, 80, chanDim)

	# four Inception modules followed by a downsample module
	x = inception_module(x, 112, 48, chanDim)
	x = inception_module(x, 96, 64, chanDim)
	x = inception_module(x, 80, 80, chanDim)
	x = inception_module(x, 48, 96, chanDim)
	x = downsample_module(x, 96, chanDim)

	# two Inception modules followed by global POOL and dropout
	x = inception_module(x, 176, 160, chanDim)
	x = inception_module(x, 176, 160, chanDim)
	x = AveragePooling2D((7, 7))(x)
	x = Dropout(0.5)(x)

	# softmax classifier
	x = Flatten()(x)
	x = Dense(classes)(x)
	x = Activation("softmax")(x)

	# create the model
	model = Model(inputs, x, name="minigooglenet")

	# return the constructed network architecture
	return model

class MiniVGGNetModel(Model):
	def __init__(self, classes, chanDim=-1):
		# call the parent constructor
		super(MiniVGGNetModel, self).__init__()

		# initialize the layers in the first (CONV => RELU) * 2 => POOL
		# layer set
		self.conv1A = Conv2D(32, (3, 3), padding="same")
		self.act1A = Activation("relu")
		self.bn1A = BatchNormalization(axis=chanDim)
		self.conv1B = Conv2D(32, (3, 3), padding="same")
		self.act1B = Activation("relu")
		self.bn1B = BatchNormalization(axis=chanDim)
		self.pool1 = MaxPooling2D(pool_size=(2, 2))

		# initialize the layers in the second (CONV => RELU) * 2 => POOL
		# layer set
		self.conv2A = Conv2D(32, (3, 3), padding="same")
		self.act2A = Activation("relu")
		self.bn2A = BatchNormalization(axis=chanDim)
		self.conv2B = Conv2D(32, (3, 3), padding="same")
		self.act2B = Activation("relu")
		self.bn2B = BatchNormalization(axis=chanDim)
		self.pool2 = MaxPooling2D(pool_size=(2, 2))

		# initialize the layers in our fully-connected layer set
		self.flatten = Flatten()
		self.dense3 = Dense(512)
		self.act3 = Activation("relu")
		self.bn3 = BatchNormalization()
		self.do3 = Dropout(0.5)

		# initialize the layers in the softmax classifier layer set
		self.dense4 = Dense(classes)
		self.softmax = Activation("softmax")

	def call(self, inputs):
		# build the first (CONV => RELU) * 2 => POOL layer set
		x = self.conv1A(inputs)
		x = self.act1A(x)
		x = self.bn1A(x)
		x = self.conv1B(x)
		x = self.act1B(x)
		x = self.bn1B(x)
		x = self.pool1(x)

		# build the second (CONV => RELU) * 2 => POOL layer set
		x = self.conv2A(x)
		x = self.act2A(x)
		x = self.bn2A(x)
		x = self.conv2B(x)
		x = self.act2B(x)
		x = self.bn2B(x)
		x = self.pool2(x)

		# build our FC layer set
		x = self.flatten(x)
		x = self.dense3(x)
		x = self.act3(x)
		x = self.bn3(x)
		x = self.do3(x)

		# build the softmax classifier
		x = self.dense4(x)
		x = self.softmax(x)

		# return the constructed model
		return x