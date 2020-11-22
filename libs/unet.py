from keras.models import *
from keras.layers import *

from .resnet50 import get_resnet50_encoder

MERGE_AXIS = -1

def unet_mini(n_classes, input_height=360, input_width=480):

    img_input = Input(shape=(input_height, input_width, 3))

    conv1 = Conv2D(32, (3, 3),
                   activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3),
                   activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3),
                   activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3),
                   activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3),
                   activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3),
                   activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(
        conv3), conv2], axis=MERGE_AXIS)
    conv4 = Conv2D(64, (3, 3),
                   activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3),
                   activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(
        conv4), conv1], axis=MERGE_AXIS)
    conv5 = Conv2D(32, (3, 3),
                   activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3),
                   activation='relu', padding='same')(conv5)

    o = Conv2D(n_classes, (1, 1),
               padding='same')(conv5)

    o = Activation('softmax')(o)

    model = Model(img_input,o)
    model.model_name = "unet_mini"
    return model


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(1024, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)
    o = (LeakyReLU(0.2)(o))

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f4], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)
    o = (LeakyReLU(0.2)(o))

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3] , axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)
    o = (LeakyReLU(0.2)(o))

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)
    o = (LeakyReLU(0.2)(o))

    o = (UpSampling2D((2, 2)))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (LeakyReLU(0.2)(o))

    o = UpSampling2D((2, 2))(o)


    o = Conv2D(n_classes, (3, 3), padding='same')(o)

    o = Activation('softmax')(o)

    model = Model(img_input,o)

    return model


# def unet(n_classes, input_height=416, input_width=608, encoder_level=3):

#     model = _unet(n_classes, vanilla_encoder,
#                   input_height=input_height, input_width=input_width)
#     model.model_name = "unet"
#     return model


# def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3):

#     model = _unet(n_classes, get_vgg_encoder,
#                   input_height=input_height, input_width=input_width)
#     model.model_name = "vgg_unet"
#     return model


def resnet50_unet(n_classes, input_height=416, input_width=608,
                  encoder_level=3):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "resnet50_unet"
    return model


# def mobilenet_unet(n_classes, input_height=224, input_width=224,
#                    encoder_level=3):

#     model = _unet(n_classes, get_mobilenet_encoder,
#                   input_height=input_height, input_width=input_width)
#     model.model_name = "mobilenet_unet"
#     return model
