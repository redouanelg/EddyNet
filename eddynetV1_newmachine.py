#
"""
Created on Wed Feb  1 13:09:44 2017

@author: Redouane Lguensat

Eddynet: A convolutional encoder-decoder for the pixel-wise segmentation of oceanic eddies from AVISO SSH MAPS

(code is based on Keras 2.0, theano dim oredering and tensorflow backend)

"""

import os
os.environ["CUDA_VISIBLE_DEVICE"]=''

############################################# Imports
from keras.models import Model, load_model
from keras.layers.core import Activation, Reshape, Permute
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, AlphaDropout, concatenate, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
#from keras_tqdm import TQDMCallback

import matplotlib.pyplot as plt
import numpy as np
import pickle

################################################################### READ SAVED DATA

# Load Segmentation maps training data (from 1998 to 2011)
#    0: no eddy
#    1: anticyclonic eddy
#    2: cyclonic eddy
#    3: land or no data (here in this work we will consider further in the code that class 3 belongs to class 0, so we only consider 3 classes)
SegmaskTot=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/SegmaskTot_19982011.npy')
# load SSH AVISO maps data 
SSH_aviso_train=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/SSH_aviso_1998_2011.npy')

################################################################### Architecture

K.set_image_dim_ordering('th') # Theano dimension ordering in this code

width = 128
height = 128
nbClass = 3

kernel = 3

###################################### INPUT LAYER

img_input = Input(shape=(1, height, width))

######################################ENCODER

conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(img_input)
conv1 = BatchNormalization(axis=1)(conv1)
conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(conv1)
conv1 = BatchNormalization(axis=1)(conv1)
conv1 = Dropout(0.25)(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(pool1)
conv2 = BatchNormalization(axis=1)(conv2)
conv2 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(conv2)
conv2 = BatchNormalization(axis=1)(conv2)
conv2 = Dropout(0.25)(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(pool2)
conv3 = BatchNormalization(axis=1)(conv3)
conv3 = Dropout(0.25)(conv3)
conv3 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(conv3)
conv3 = BatchNormalization(axis=1)(conv3)
conv3 = Dropout(0.25)(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#######################################center

convC = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(pool3)
convC = BatchNormalization(axis=1)(convC)
convC = Conv2D(32, (kernel, kernel), padding="same", activation='relu' , kernel_initializer='he_normal')(convC)
convC = BatchNormalization(axis=1)(convC)
convC = Dropout(0.25)(convC)

#######################################DECODER

up3 = concatenate([Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal' )(convC), conv3], axis=1)
decod3 = BatchNormalization(axis=1)(up3)
decod3 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod3)
decod3 = BatchNormalization(axis=1)(decod3)
decod3 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod3)
decod3 = BatchNormalization(axis=1)(decod3)
decod3 = Dropout(0.25)(decod3)

up2 = concatenate([Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal' )(decod3), conv2], axis=1)
decod2 = BatchNormalization(axis=1)(up2)
decod2 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod2)
decod2 = BatchNormalization(axis=1)(decod2)
decod2 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod2)
decod2 = BatchNormalization(axis=1)(decod2)
decod2 = Dropout(0.25)(decod2)

up1 = concatenate([Conv2DTranspose(32, (kernel, kernel), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal' )(decod2), conv1], axis=1)
decod1 = BatchNormalization(axis=1)(up1)
decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod1)
decod1 = BatchNormalization(axis=1)(decod1)
decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu', kernel_initializer='he_normal' )(decod1)
decod1 = BatchNormalization(axis=1)(decod1)
decod1 = Dropout(0.25)(decod1)

####################################### Segmentation Layer

x = Conv2D(nbClass, (1, 1), padding="valid" )(decod1) 
x = Reshape((nbClass, height * width))(x) 
x = Permute((2, 1))(x)
x = Activation("softmax")(x)
eddynet = Model(img_input, x)

############################################################################################# LOSS
smooth = 1e-5

def dice_coef_anti(y_true, y_pred):
    y_true_anti = y_true[:,:,1]
    y_pred_anti = y_pred[:,:,1]
    intersection_anti = K.sum(y_true_anti * y_pred_anti)
    return (2 * intersection_anti + smooth) / (K.sum(y_true_anti)+ K.sum(y_pred_anti) + smooth)

def dice_coef_cyc(y_true, y_pred):
    y_true_cyc = y_true[:,:,2]
    y_pred_cyc = y_pred[:,:,2]
    intersection_cyc = K.sum(y_true_cyc * y_pred_cyc)
    return (2 * intersection_cyc + smooth) / (K.sum(y_true_cyc) + K.sum(y_pred_cyc) + smooth)

def dice_coef_nn(y_true, y_pred):
    y_true_nn = y_true[:,:,0]
    y_pred_nn = y_pred[:,:,0]
    intersection_nn = K.sum(y_true_nn * y_pred_nn)
    return (2 * intersection_nn + smooth) / (K.sum(y_true_nn) + K.sum(y_pred_nn) + smooth)
    
def mean_dice_coef(y_true, y_pred):
    return (dice_coef_anti(y_true, y_pred) + dice_coef_cyc(y_true, y_pred) + dice_coef_nn(y_true, y_pred))/3.
           
def dice_coef_loss(y_true, y_pred):
    return 1 - mean_dice_coef(y_true, y_pred)    

############################################################################################# COMPILE
   
eddynet.compile(optimizer='adam', loss=dice_coef_loss,
                metrics=['categorical_accuracy', mean_dice_coef])

#eddynet.compile(optimizer='adam', loss='categorical_crossentropy',
#                metrics=['categorical_accuracy', mean_dice_coef])
                
from keras.utils import plot_model
plot_model(eddynet, to_file='eddynet.png')

################################################ Train/Test data (PATCH version)
#
from sklearn.feature_extraction.image import extract_patches_2d

pheight=128
pwidth=128
max_patches=1 # number of patches for each day
random_state=555 #for reproductivity

window_shape = (pheight, pwidth)
nbdaysTrain=5100 #number of considered days from the available training days
strideDays=1 #strideDays*nbdaysTrain should be less than the number of the available training images here 5116
nbpatchs=max_patches

x_train=np.zeros((nbdaysTrain*nbpatchs, 1, pheight, pwidth))
BB_label=np.zeros((nbdaysTrain*nbpatchs,pheight,pwidth)).astype(int)

for dayN in range(nbdaysTrain):
    x_train[dayN*nbpatchs:dayN*nbpatchs+nbpatchs,0,:,:] = extract_patches_2d(SSH_aviso_train[0:128,:,strideDays*dayN+1], window_shape, max_patches, random_state+dayN)
    BB_label[dayN*nbpatchs:dayN*nbpatchs+nbpatchs,:,:] = extract_patches_2d(SegmaskTot[0:128,:,strideDays*dayN], window_shape, max_patches, random_state+dayN)

###   
x_train[x_train<-100]=0. ##### change the standard AVISO fill value to zero
BB_label[BB_label==3]=0. ##### class 3 is class 0 in this work

label_train=np.reshape(BB_label, (len(x_train), 1, pheight*pwidth)).transpose((0,2,1))
x_train_label=np.zeros((len(x_train),pheight*pwidth,nbClass))
for kk in range(len(x_train)):
    print kk
    x_train_label[kk,:,:] = np_utils.to_categorical(label_train[kk,:,:],nbClass)
    
### memory cleaning  
del SegmaskTot, SSH_aviso_train
 
### memory cleaning  
#del BB_label 
 
############################################### EDDYNET
filepath="weights-minvaloss-eddynetpaper32BNRDropout_final.h5"
#filepath="eddynet32SELU_alphaDropout005_bnafterconv2transp.h5"

from keras.callbacks import EarlyStopping, ModelCheckpoint, History
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
eddynethistory = History()
callbacks_list = [early_stopping,checkpoint,eddynethistory]  
          
eddyhist=eddynet.fit(x_train, x_train_label,
                        epochs=100,
                        batch_size=16, #32
                        shuffle=True,
                        validation_split=0.2,
                        verbose=2,
                        callbacks=callbacks_list)                
                                
# eddynet.save('Eddynetpaper64BNRnoDrop.h5')
                                
with open('History32RBNlekher.pickle', 'wb') as file_pi:
    pickle.dump(eddynet.history.history, file_pi)
          
## load model
#eddynet = load_model('weights-minvalloss-eddynetpaper32BNRDropout_random2.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'mean_dice_coef': mean_dice_coef})

########################################### AVISO TEST 2012

SSH_aviso_2012=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/SSH_aviso_2012.npy')
SSH_aviso_2012[SSH_aviso_2012<-100]=0.

Segmask2012=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/segmask2012.npy')
Segmask2012[Segmask2012==3]=0.

ghostAnti=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/Antighostpositions2012.npy')
ghostCyc=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/Cycghostpositions2012.npy')

########################################### extract random 2012 SSH patches
nbdaysfrom2012=360
strideDays2012=1
nbpatchs2012=1
random_state2=555
mdice=[]
mdiceanticyc=[]
mdicecyc=[]
mdicenoneddy=[]
acc=[]
#150:-1
for w in range(50):
    Test2012=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth))
    Test2012_label=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth))#.astype(int)
    #GhostAnticenters=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth))
    #GhostCyccenters=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth))
    
    for dayN in range(nbdaysfrom2012):
        Test2012[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = extract_patches_2d(SSH_aviso_2012[0:128,:,strideDays2012*dayN+1], window_shape, nbpatchs2012, w*(random_state2+dayN))
        Test2012_label[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = extract_patches_2d(Segmask2012[0:128,:,strideDays2012*dayN], window_shape, nbpatchs2012, w*(random_state2+dayN))
        #GhostAnticenters[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = extract_patches_2d(ghostAnti[0:128,:,strideDays2012*dayN], window_shape, nbpatchs2012, random_state2)
        #GhostCyccenters[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = extract_patches_2d(ghostCyc[0:128,:,strideDays2012*dayN], window_shape, nbpatchs2012, random_state2)
    
    #del SSH_aviso_2012, Segmask2012
    ################ scores for test
    
    #scores_2012 = eddynet.evaluate(Test2012[:,None,:,:], Test2012_label, batch_size=3)
    
    predict2012=eddynet.predict(Test2012[:,None,:,:],batch_size=1)
    predict2012Im=np.reshape(predict2012.argmax(2),(nbdaysfrom2012*nbpatchs2012,pheight,pwidth))
    interanticyc=np.logical_and(predict2012Im==1,Test2012_label==1).sum()
    intercyc=np.logical_and(predict2012Im==2,Test2012_label==2).sum()
    internoneddy=np.logical_and(predict2012Im==0,Test2012_label==0).sum()
    diceanticyc=(interanticyc*2.)/((predict2012Im==1).sum()+(Test2012_label==1).sum())
    dicecyc=(intercyc*2.)/((predict2012Im==2).sum()+(Test2012_label==2).sum())
    dicenoneddy=(internoneddy*2.)/((predict2012Im==0).sum()+(Test2012_label==0).sum())
    mdiceanticyc.append(diceanticyc)
    mdicecyc.append(dicecyc)
    mdicenoneddy.append(dicenoneddy)
    mdice.append(np.mean([dicecyc,diceanticyc,dicenoneddy]))
    #print mdice
    
    acc.append((1.*(interanticyc+intercyc+internoneddy))/Test2012_label.size)
    #print acc

print np.mean(mdiceanticyc),np.std(mdiceanticyc)
print np.mean(mdicecyc),np.std(mdicecyc)
print np.mean(mdicenoneddy),np.std(mdicenoneddy)
print np.mean(mdice),np.std(mdice)
print np.mean(acc),np.std(acc)
#print("Mean Dice score is %f(%f)", np.mean(mdice),np.std(mdice))
############################ Patch plot
plt.figure()
plt.ion()
#randpatches=np.random.randint(0,nbdaysfrom2012*nbpatchs2012,20)
randpatches=range(50)

for i in randpatches:
    plt.clf()
    # Ghost centers
    #ghostAntilat,ghostAntilon=np.where(GhostAnticenters[i,:,:]==1)
    #ghostCyclat,ghostCyclon=np.where(GhostCyccenters[i,:,:]==2)
    # display original
    ax = plt.subplot(1, 3, 1)
    plt.imshow(Test2012[i,:,:])
    #plt.scatter(ghostAntilon,ghostAntilat,c="r")
    #plt.scatter(ghostCyclon,ghostCyclat,c="b")
    ax.set_title("SSH patch")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    #plt.colorbar()
    
    # display ground truth segm
    ax = plt.subplot(1, 3, 2)
    plt.imshow(Test2012_label[i,:,:])
    #plt.scatter(ghostAntilon,ghostAntilat,c="r")
    #plt.scatter(ghostCyclon,ghostCyclat,c="b")
    ax.set_title("Ground Truth segmentation")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #plt.colorbar()

    # display reconstruction
    ax = plt.subplot(1, 3, 3)
    predictSeg=eddynet.predict(np.reshape(Test2012[i,:,:],(1,1,height,width)))
    plt.imshow(np.reshape(predictSeg.argmax(2).T,(pheight,pwidth)))
    #plt.scatter(ghostAntilon,ghostAntilat,c="r")
    #plt.scatter(ghostCyclon,ghostCyclat,c="b")
    ax.set_title("EddyNet segmentation")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 
    #plt.colorbar()
    
    plt.show()
    plt.draw()
    #plt.pause(1)
    plt.waitforbuttonpress()
    
#####################  HISTORY
#with open('History32SELU_with_alphaDropout_withBnbeforetranspose.pickle', 'rb') as handle:
#     eddylosshist = pickle.load(handle) 
# summarize history for loss
plt.plot(eddynet.history.history['loss'])
plt.plot(eddynet.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.show()

plt.plot(eddyhist.history['acc'])
plt.plot(eddyhist.history['val_acc'])
plt.legend(['train', 'test'], loc='upper left')
plt.show()
##
################################## Ghost eddies
#plt.figure()
#plt.ion()
#
#for i in range(30):
#    plt.clf()
#    # Ghost centers
#    ghostAntilat1,ghostAntilon1=np.where(GhostAnticenters[i,:,:]==1)
#    ghostCyclat1,ghostCyclon1=np.where(GhostCyccenters[i,:,:]==2)
#    ghostAntilat2,ghostAntilon2=np.where(GhostAnticenters[i+1,:,:]==1)
#    ghostCyclat2,ghostCyclon2=np.where(GhostCyccenters[i+1,:,:]==2)
#    ghostAntilat3,ghostAntilon3=np.where(GhostAnticenters[i+2,:,:]==1)
#    ghostCyclat3,ghostCyclon3=np.where(GhostCyccenters[i+2,:,:]==2)
#
#    # display original
#    ax = plt.subplot(3, 3, 1)
#    plt.imshow(Test2012[i,:,:])
#    plt.scatter(ghostAntilon1,ghostAntilat1,c="r")
#    plt.scatter(ghostCyclon1,ghostCyclat1,c="b")
#    ax.set_title("SSH patch t-1")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    
#    # display original
#    ax = plt.subplot(3, 3, 2)
#    plt.imshow(Test2012[i+1,:,:])
#    plt.scatter(ghostAntilon2,ghostAntilat2,c="r")
#    plt.scatter(ghostCyclon2,ghostCyclat2,c="b")
#    ax.set_title("SSH patch t")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)    
#    
#    # display original
#    ax = plt.subplot(3, 3, 3)
#    plt.imshow(Test2012[i+2,:,:])
#    plt.scatter(ghostAntilon3,ghostAntilat3,c="r")
#    plt.scatter(ghostCyclon3,ghostCyclat3,c="b")
#    ax.set_title("SSH patch t+1")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)    
#    
#    # display ground truth segm
#    ax = plt.subplot(3, 3, 4)
#    plt.imshow(Test2012_label[i,:,:])
#    plt.scatter(ghostAntilon1,ghostAntilat1,c="r")
#    plt.scatter(ghostCyclon1,ghostCyclat1,c="b")
#    ax.set_title("Ground Truth segmentation t-1")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    
#    # display ground truth segm
#    ax = plt.subplot(3, 3, 5)
#    plt.imshow(Test2012_label[i+1,:,:])
#    plt.scatter(ghostAntilon2,ghostAntilat2,c="r")
#    plt.scatter(ghostCyclon2,ghostCyclat2,c="b")
#    ax.set_title("Ground Truth segmentation t")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    
#    # display ground truth segm
#    ax = plt.subplot(3, 3, 6)
#    plt.imshow(Test2012_label[i+2,:,:])
#    plt.scatter(ghostAntilon3,ghostAntilat3,c="r")
#    plt.scatter(ghostCyclon3,ghostCyclat3,c="b")
#    ax.set_title("Ground Truth segmentation t+1")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    #plt.colorbar()
#
#    # display reconstruction
#    ax = plt.subplot(3, 3, 7)
#    predictSeg=eddynet.predict(np.reshape(Test2012[i,:,:],(1,1,height,width)))
#    plt.imshow(np.reshape(predictSeg.argmax(2).T,(pheight,pwidth)))
#    plt.scatter(ghostAntilon1,ghostAntilat1,c="r")
#    plt.scatter(ghostCyclon1,ghostCyclat1,c="b")
#    ax.set_title("EddyNet segmentation t-1")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False) 
#
#    # display reconstruction
#    ax = plt.subplot(3, 3, 8)
#    predictSeg=eddynet.predict(np.reshape(Test2012[i+1,:,:],(1,1,height,width)))
#    plt.imshow(np.reshape(predictSeg.argmax(2).T,(pheight,pwidth)))
#    plt.scatter(ghostAntilon2,ghostAntilat2,c="r")
#    plt.scatter(ghostCyclon2,ghostCyclat2,c="b")
#    ax.set_title("EddyNet segmentation t")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False) 
#
#    # display reconstruction
#    ax = plt.subplot(3, 3, 9)
#    predictSeg=eddynet.predict(np.reshape(Test2012[i+2,:,:],(1,1,height,width)))
#    plt.imshow(np.reshape(predictSeg.argmax(2).T,(pheight,pwidth)))
#    plt.scatter(ghostAntilon3,ghostAntilat3,c="r")
#    plt.scatter(ghostCyclon3,ghostCyclat3,c="b")
#    ax.set_title("EddyNet segmentation t+1")
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False) 
#
#    plt.show()
#    plt.draw()
#    #plt.pause(1)
#    plt.waitforbuttonpress()
#    
#    
################################## plot train 


plt.figure()
plt.ion()
randpatches=np.random.randint(0,5000,20)
for i in randpatches:
    # display original
    ax = plt.subplot(1, 3, 1)
    #m_pcolor(lon_aviso,lat_aviso,SSH_aviso_2012[:,:,i+1],'ocean')
    #plt.imshow(SSH_aviso_2012[0:120,99:-1,i+1])
    plt.imshow(x_train[i,0,:,:])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display ground truth
    ax = plt.subplot(1, 3, 2)
    #m_pcolor(lon_aviso,lat_aviso,Segmask2012[:,:,i],'ocean')
    #plt.imshow(Segmask2012[0:120,99:-1,i])
    plt.imshow(BB_label[i,:,:])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display segmentation result
    ax = plt.subplot(1, 3, 3)
    #predictSeg=eddynet.predict(np.reshape(SSH_aviso_2012[0:120,99:-1,i+1],(1,1,height,width)))
    predictSeg=eddynet.predict(np.reshape(x_train[i,0,:,:],(1,1,height,width)))
    imgSeg=np.reshape(predictSeg.argmax(2).T,(height,width))
    #m_pcolor(lon_aviso,lat_aviso,imgSeg,'ocean')
    plt.imshow(imgSeg)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    plt.show()
    plt.draw()
    #plt.pause(1)
    plt.waitforbuttonpress()
##############################
    
    
#### Class weights     
#labels_dict=[sum(sum(label_train==i))[0] for i in range(nbClass)]
#            
#del label_train
#
##from sklearn.utils import compute_class_weight
##class_weight=compute_class_weight('balanced',np.unique(label_train),label_train.flatten())
#CW=[np.sum(labels_dict) / float((nbClass * labels_dict[i])) for i in range(nbClass)] #scikitlearn
##
##freqs=[labels_dict[i]/float(np.sum(labels_dict))  for i in range(nbClass)]
##CW=[np.median(freqs)/freqs[i]  for i in range(nbClass)]
###
#CW=[1.,20.,34.]
#sample_weight=np.reshape(BB_label, (len(BB_label), pheight*pwidth)).copy().astype('float32')
## warning: pay attention if one of your weights is 0,1,2 or 3
#sample_weight[sample_weight==1]=CW[1]
#sample_weight[sample_weight==0]=CW[0]  #important to start with 1  before 0 class_weight[0] is equal to 1
#sample_weight[sample_weight==2]=CW[2]

##################################
#def jaccard_loss(y_true, y_pred):
#    return 1 - mean_jaccard_coef(y_true, y_pred)
#
#def dice_coef_hard_anti(y_true, y_pred):
#    y_true_anti = y_true[:,:,1]
#    y_pred_anti = K.round(y_pred[:,:,1])
#    intersection_anti = K.sum(y_true_anti * y_pred_anti)
#    return (2 * intersection_anti + smooth) / (K.sum(y_true_anti)+ K.sum(y_pred_anti) + smooth)
#
#def dice_coef_hard_cyc(y_true, y_pred):
#    y_true_cyc = y_true[:,:,2]
#    y_pred_cyc = K.round(y_pred[:,:,2])
#    intersection_cyc = K.sum(y_true_cyc * y_pred_cyc)
#    return (2 * intersection_cyc + smooth) / (K.sum(y_true_cyc) + K.sum(y_pred_cyc) + smooth)
#
#def dice_coef_hard_nn(y_true, y_pred):
#    y_true_nn = y_true[:,:,0]
#    y_pred_nn = K.round(y_pred[:,:,0])
#    intersection_nn = K.sum(y_true_nn * y_pred_nn)
#    return (2 * intersection_nn + smooth) / (K.sum(y_true_nn) + K.sum(y_pred_nn) + smooth)
#    
#def mean_dice_coef_hard(y_true, y_pred):
#    return (dice_coef_hard_cyc(y_true, y_pred)+dice_coef_hard_nn(y_true, y_pred)+dice_coef_hard_anti(y_true, y_pred))/3.