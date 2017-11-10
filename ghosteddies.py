# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:30:57 2017

@author: rlguensa
"""

############################################# Imports
from keras.models import load_model
from eddynetV1_newmachine import dice_coef_loss,mean_dice_coef
import matplotlib.pyplot as plt
import numpy as np

pheight,pwidth=128,128


## load model
eddynet = load_model('weights-minvaloss-eddynetpaper32BNRDropout_final.h5', \
                     custom_objects={'dice_coef_loss': dice_coef_loss, \
                     'mean_dice_coef': mean_dice_coef})

########################################### AVISO TEST 2012

SSH_aviso_2012=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/SSH_aviso_2012.npy')
SSH_aviso_2012[SSH_aviso_2012<-100]=0.

Segmask2012=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/segmask2012.npy')
Segmask2012[Segmask2012==3]=0.

ghostAnti=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/Antighostpositions2012.npy')
ghostCyc=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/Evandata/Cycghostpositions2012.npy')

########################################### extract SSH patches
nbdaysfrom2012=360
strideDays2012=1
nbpatchs2012=1
random_state2=333
#150:-1
Test2012=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth))
Test2012_label=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth))#.astype(int)
GhostAnticenters=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth))
GhostCyccenters=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth))

for dayN in range(nbdaysfrom2012):
    Test2012[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = SSH_aviso_2012[0:128,151:-1,strideDays2012*dayN+1]
    Test2012_label[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = Segmask2012[0:128,151:-1,strideDays2012*dayN]
    GhostAnticenters[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = ghostAnti[0:128,151:-1,strideDays2012*dayN]
    GhostCyccenters[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = ghostCyc[0:128,151:-1,strideDays2012*dayN]

del SSH_aviso_2012, Segmask2012
############################ Patch plot
plt.figure()
plt.ion()

for i in range(300,nbdaysfrom2012):
    plt.clf()
    print i
    # Ghost centers
    ghostAntilat,ghostAntilon=np.where(GhostAnticenters[i,:,:]==1)
    ghostCyclat,ghostCyclon=np.where(GhostCyccenters[i,:,:]==2)
    # display original
    ax = plt.subplot(1, 3, 1)
    plt.imshow(Test2012[i,:,:])
    plt.scatter(ghostAntilon,ghostAntilat,c="r")
    plt.scatter(ghostCyclon,ghostCyclat,c="b")
    ax.set_title("SSH patch")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    #plt.colorbar()
    
    # display ground truth segm
    ax = plt.subplot(1, 3, 2)
    plt.imshow(Test2012_label[i,:,:])
    plt.scatter(ghostAntilon,ghostAntilat,c="r")
    plt.scatter(ghostCyclon,ghostCyclat,c="b")
    ax.set_title("Ground Truth segmentation")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #plt.colorbar()

    # display reconstruction
    ax = plt.subplot(1, 3, 3)
    predictSeg=eddynet.predict(np.reshape(Test2012[i,:,:],(1,1,pheight,pwidth)))
    plt.imshow(np.reshape(predictSeg.argmax(2).T,(pheight,pwidth)))
    plt.scatter(ghostAntilon,ghostAntilat,c="r")
    plt.scatter(ghostCyclon,ghostCyclat,c="b")
    ax.set_title("EddyNet segmentation")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 
    #plt.colorbar()
    
    plt.show()
    plt.draw()
    #plt.pause(0.1)
    plt.waitforbuttonpress()
    
################ scores for test

#scores_2012 = eddynet.evaluate(Test2012[:,None,:,:], Test2012_label, batch_size=3)

predict2012=eddynet.predict(Test2012[:,None,:,:], batch_size=1)
predict2012Im=np.reshape(predict2012.argmax(2),(nbdaysfrom2012*nbpatchs2012,pheight,pwidth))
ghostanticyc=np.logical_and(GhostAnticenters==1,predict2012Im==1).sum()
ghostcyc=np.logical_and(GhostCyccenters==2,predict2012Im==2).sum()

GhostAntidetected=(1.*ghostanticyc)/((GhostAnticenters==1).sum())
GhostCycdetected=(1.*ghostcyc)/((GhostCyccenters==2).sum())

print GhostAntidetected,GhostCycdetected

################################# Ghost eddies
plt.figure()
plt.ion()

for i in range(30):
    plt.clf()
    # Ghost centers
    ghostAntilat1,ghostAntilon1=np.where(GhostAnticenters[i,:,:]==1)
    ghostCyclat1,ghostCyclon1=np.where(GhostCyccenters[i,:,:]==2)
    ghostAntilat2,ghostAntilon2=np.where(GhostAnticenters[i+1,:,:]==1)
    ghostCyclat2,ghostCyclon2=np.where(GhostCyccenters[i+1,:,:]==2)
    ghostAntilat3,ghostAntilon3=np.where(GhostAnticenters[i+2,:,:]==1)
    ghostCyclat3,ghostCyclon3=np.where(GhostCyccenters[i+2,:,:]==2)

    # display original
    ax = plt.subplot(3, 3, 1)
    plt.imshow(Test2012[i,:,:])
    plt.scatter(ghostAntilon1,ghostAntilat1,c="r")
    plt.scatter(ghostCyclon1,ghostCyclat1,c="b")
    ax.set_title("SSH patch t-1")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display original
    ax = plt.subplot(3, 3, 2)
    plt.imshow(Test2012[i+1,:,:])
    plt.scatter(ghostAntilon2,ghostAntilat2,c="r")
    plt.scatter(ghostCyclon2,ghostCyclat2,c="b")
    ax.set_title("SSH patch t")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    
    # display original
    ax = plt.subplot(3, 3, 3)
    plt.imshow(Test2012[i+2,:,:])
    plt.scatter(ghostAntilon3,ghostAntilat3,c="r")
    plt.scatter(ghostCyclon3,ghostCyclat3,c="b")
    ax.set_title("SSH patch t+1")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    
    # display ground truth segm
    ax = plt.subplot(3, 3, 4)
    plt.imshow(Test2012_label[i,:,:])
    plt.scatter(ghostAntilon1,ghostAntilat1,c="r")
    plt.scatter(ghostCyclon1,ghostCyclat1,c="b")
    ax.set_title("Ground Truth segmentation t-1")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display ground truth segm
    ax = plt.subplot(3, 3, 5)
    plt.imshow(Test2012_label[i+1,:,:])
    plt.scatter(ghostAntilon2,ghostAntilat2,c="r")
    plt.scatter(ghostCyclon2,ghostCyclat2,c="b")
    ax.set_title("Ground Truth segmentation t")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display ground truth segm
    ax = plt.subplot(3, 3, 6)
    plt.imshow(Test2012_label[i+2,:,:])
    plt.scatter(ghostAntilon3,ghostAntilat3,c="r")
    plt.scatter(ghostCyclon3,ghostCyclat3,c="b")
    ax.set_title("Ground Truth segmentation t+1")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #plt.colorbar()

    # display reconstruction
    ax = plt.subplot(3, 3, 7)
    predictSeg=eddynet.predict(np.reshape(Test2012[i,:,:],(1,1,pheight,pwidth)))
    plt.imshow(np.reshape(predictSeg.argmax(2).T,(pheight,pwidth)))
    plt.scatter(ghostAntilon1,ghostAntilat1,c="r")
    plt.scatter(ghostCyclon1,ghostCyclat1,c="b")
    ax.set_title("EddyNet segmentation t-1")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 

    # display reconstruction
    ax = plt.subplot(3, 3, 8)
    predictSeg=eddynet.predict(np.reshape(Test2012[i+1,:,:],(1,1,pheight,pwidth)))
    plt.imshow(np.reshape(predictSeg.argmax(2).T,(pheight,pwidth)))
    plt.scatter(ghostAntilon2,ghostAntilat2,c="r")
    plt.scatter(ghostCyclon2,ghostCyclat2,c="b")
    ax.set_title("EddyNet segmentation t")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 

    # display reconstruction
    ax = plt.subplot(3, 3, 9)
    predictSeg=eddynet.predict(np.reshape(Test2012[i+2,:,:],(1,1,pheight,pwidth)))
    plt.imshow(np.reshape(predictSeg.argmax(2).T,(pheight,pwidth)))
    plt.scatter(ghostAntilon3,ghostAntilat3,c="r")
    plt.scatter(ghostCyclon3,ghostCyclat3,c="b")
    ax.set_title("EddyNet segmentation t+1")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 

    plt.show()
    plt.draw()
    #plt.pause(1)
    plt.waitforbuttonpress()
    
    