import tensorflow as tf 
from keras import layers, models
import numpy as np
from .resnet50 import get_resnet50_encoder
from .vgg16 import get_vgg_encoder

def segnet_decoder(f, n_classes, n_up=3):

	assert n_up >= 2

	o = f
	o = (layers.ZeroPadding2D((1, 1)))(o)
	o = (layers.Conv2D(512, (3, 3), padding='valid'))(o)
	o = (layers.BatchNormalization())(o)
	o = (layers.LeakyReLU(0.2)(o))
	# o = (layers.ZeroPadding2D((1, 1)))(o)
	# o = (layers.Conv2D(512, (3, 3), padding='valid'))(o)
	# o = (layers.BatchNormalization())(o)

	# o = (layers.ZeroPadding2D((1, 1)))(o)
	# o = (layers.Conv2D(512, (3, 3), padding='valid'))(o)
	# o = (layers.BatchNormalization())(o)

	# o = (layers.UpSampling2D((2, 2)))(o)
	# o = (layers.ZeroPadding2D((1, 1)))(o)
	# o = (layers.Conv2D(256, (3, 3), padding='valid'))(o)
	# o = (layers.BatchNormalization())(o)

	o = (layers.UpSampling2D((2, 2)))(o)
	o = (layers.ZeroPadding2D((1, 1)))(o)
	o = (layers.Conv2D(256, (3, 3), padding='valid'))(o)
	o = (layers.BatchNormalization())(o)
	o = (layers.LeakyReLU(0.2)(o))

	for _ in range(n_up-2):
		o = (layers.UpSampling2D((2, 2)))(o)
		o = (layers.ZeroPadding2D((1, 1)))(o)
		o = (layers.Conv2D(128, (3, 3), padding='valid'))(o)
		o = (layers.BatchNormalization())(o)
		o = (layers.LeakyReLU(0.2)(o))

	o = (layers.UpSampling2D((2, 2)))(o)
	o = (layers.ZeroPadding2D((1, 1)))(o)
	o = (layers.Conv2D(64, (3, 3), padding='valid'))(o)
	o = (layers.BatchNormalization())(o)
	o = (layers.LeakyReLU(0.2)(o))

	# o = (layers.UpSampling2D((2, 2)))(o)
	# o = (layers.ZeroPadding2D((1, 1)))(o)
	# o = (layers.Conv2D(64, (3, 3), padding='valid'))(o)
	# o = (layers.BatchNormalization())(o)

	o = layers.Conv2D(n_classes, (3, 3), padding='same')(o)


	return o

def _segnet(n_classes, encoder,  input_height=416, input_width=608,
			encoder_level=6):

	img_input, feat = encoder(
		input_height=input_height,  input_width=input_width)

	# feat = levels[encoder_level]
	print(feat)
	o = segnet_decoder(feat, n_classes, n_up=5)
	o = (layers.Activation('softmax'))(o)
	model = models.Model(img_input, o)

	return model

def resnet50_segnet(n_classes, input_height=416, input_width=608,
					encoder_level=3):

	model = _segnet(n_classes, get_resnet50_encoder, input_height=input_height,
					input_width=input_width, encoder_level=encoder_level)
	return model

def vgg16_segnet(n_classes, input_height=416, input_width=608,
					encoder_level=3):

	model = _segnet(n_classes, get_vgg_encoder, input_height=input_height,
					input_width=input_width, encoder_level=encoder_level)
	return model

def build_unet(size=300, basef=64, maxf=512, encoder='resnet50', pretrained=True):
	input = layers.Input((size, size, 3))

	encoder_model = make_encoder(input, name=encoder, pretrained=pretrained)
	print(encoder_model)
	crosses = []

	for layer in encoder_model.layers:
		# don't end on padding layers
		if type(layer) == layers.ZeroPadding2D:
			continue
		# print("Layer")
		# print(layer.output_shape[0][1])
		
		if len(layer.output_shape) > 1:
			idx = get_scale_index(size, layer.output_shape[1])
		else:
			idx = get_scale_index(size, layer.output_shape[0][1])
		if idx is None:
			continue
		if idx >= len(crosses):
			crosses.append(layer)
		else:
			crosses[idx] = layer

	x = crosses[-1].output
	for scale in range(len(crosses)-2, -1, -1):
		nf = min(basef * 2**scale, maxf)
		x = upscale(x, nf)
		x = act(x)
		x = layers.Concatenate()([
			pad_to_scale(x, scale, size=size),
			pad_to_scale(crosses[scale].output, scale, size=size)
		])
		x = conv(x, nf)
		x = act(x)

	x = conv(x, 4)
	x = layers.Activation('softmax')(x)

	return models.Model(input, x)

def make_encoder(input, name='resnet50', pretrained=True):
	if name == 'resnet18':
		from classification_models.keras import Classifiers
		ResNet18, _ = Classifiers.get('resnet18')
		model = ResNet18(
			weights='imagenet' if pretrained else None,
			input_tensor=input,
			include_top=False
		)
	elif name == 'resnet50':
		from keras.applications.resnet import ResNet50
		model = ResNet50(
			weights='imagenet' if pretrained else None,
			input_tensor=input,
			include_top=False
		)
	elif name == 'resnet101':
		from keras.applications.resnet import ResNet101
		model = ResNet101(
			weights='imagenet' if pretrained else None,
			input_tensor=input,
			include_top=False
		)
	elif name == 'resnet152':
		from keras.applications.resnet import ResNet152
		model = ResNet152(
			weights='imagenet' if pretrained else None,
			input_tensor=input,
			include_top=False
		)
	elif name == 'vgg16':
		from keras.applications.vgg16 import VGG16
		model = VGG16(
			weights='imagenet' if pretrained else None,
			input_tensor=input,
			include_top=False
		)
	elif name == 'vgg19':
		from keras.applications.vgg19 import VGG19
		model = VGG19(
			weights='imagenet' if pretrained else None,
			input_tensor=input,
			include_top=False
		)
	else:
		raise Exception(f'unknown encoder {name}')

	return model

def get_scale_index(in_size, l_size):
	for i in range(8):
		s_size = in_size // (2 ** i)
		if abs(l_size - s_size) <= 4:
			return i
	return None

def pad_to_scale(x, scale, size=300):
	expected = int(np.ceil(size / (2. ** scale)))
	diff = expected - int(x.shape[1])
	if diff > 0:
		left = diff // 2
		right = diff - left
		x = reflectpad(x, (left, right))
	elif diff < 0:
		left = -diff // 2
		right = -diff - left
		x = layers.Cropping2D(((left, right), (left, right)))(x)
	return x

def reflectpad(x, pad):
	return layers.Lambda(lambda x: tf.pad(x, [(0, 0), pad, pad, (0, 0)], 'REFLECT'))(x)

def upscale(x, nf):
	x = layers.UpSampling2D((2, 2))(x)
	x = conv(x, nf, kernel_size=(1, 1))
	return x

def act(x):
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU(0.2)(x)
	return x

def conv(x, nf, kernel_size=(3, 3), **kwargs):
	padleft = (kernel_size[0] - 1) // 2
	padright = kernel_size[0] - 1 - padleft
	if padleft > 0 or padright > 0:
		x = reflectpad(x, (padleft, padright))
	return layers.Conv2D(nf, kernel_size=kernel_size, padding='valid', **kwargs)(x)
