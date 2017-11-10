# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:04:20 2017

@author: rlguensa
"""


###################################### INPUT LAYER

img_input = Input(shape=(1, height, width))

######################################ENCODER

conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(img_input)
conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(conv1)
conv1 = AlphaDropout(0.05)(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
pool1 = BatchNormalization(axis=1)(pool1)

conv2 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(pool1)
conv2 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(conv2)
conv2 = AlphaDropout(0.05)(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
pool2 = BatchNormalization(axis=1)(pool2)

conv3 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(pool2)
conv3 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(conv3)
conv3 = AlphaDropout(0.05)(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
pool3 = BatchNormalization(axis=1)(pool3)

#######################################center

convC = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(pool3)
convC = Conv2D(32, (kernel, kernel), padding="same", activation='selu' , kernel_initializer='lecun_normal')(convC)
convC = AlphaDropout(0.05)(convC)


#######################################DECODER

convTrans3 = Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='selu', kernel_initializer='lecun_normal' )(convC)
convTrans3 = BatchNormalization(axis=1)(convTrans3)
up3 = concatenate([convTrans3, conv3], axis=1)
decod3 = BatchNormalization(axis=1)(up3)
decod3 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(decod3)
decod3 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(decod3)
decod3 = AlphaDropout(0.05)(decod3)


convTrans2 = Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='selu', kernel_initializer='lecun_normal' )(decod3)
convTrans2 = BatchNormalization(axis=1)(convTrans2)
up2 = concatenate([convTrans2, conv2], axis=1)
decod2 = BatchNormalization(axis=1)(up2)
decod2 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(decod2)
decod2 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(decod2)
decod2 = AlphaDropout(0.05)(decod2)


convTrans1 = Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='selu', kernel_initializer='lecun_normal' )(decod2)
convTrans1 = BatchNormalization(axis=1)(convTrans1)
up1 = concatenate([convTrans1, conv1], axis=1)
decod1 = BatchNormalization(axis=1)(up1)
decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(decod1)
decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='selu', kernel_initializer='lecun_normal' )(decod1)

####################################### Segmentation Layer

x = Conv2D(nbClass, (1, 1), padding="valid" )(decod1) 
x = Reshape((nbClass, height * width))(x) 
x = Permute((2, 1))(x)
x = Activation("softmax")(x)
eddynet = Model(img_input, x)
