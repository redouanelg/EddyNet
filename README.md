# EddyNet
EddyNet: A Deep Neural Network For Pixel-Wise Classification of Oceanic Eddies

This is the supplementary material of the publication "EddyNet: A Deep Neural Network For Pixel-Wise Classification of Oceanic Eddies
", from R. Lguensat et al., submitted. Pre-print at: 

Eddynet is an U-Net like architecture (a convolutional encoder-decoder followed by a pixel-wise classification layer + skip connections). <br />

![](unetschema.png)

# Paper main messages:
> A deep neural net that "emulates" the result of a geometry based and expert based method 
> Comparing EddyNet with a version where we use SELU activation function (EddyNet_S). Replacing directly ReLU+BN with SELU resulted in a noisy loss and hurted the performance, we then kept BN after maxpooling, transposed deconvolution and concatenation.
> For this multiclass classification problem, we use (1-mean dice coefficient) as a loss function instead of the categorical cross entropy loss
> Eddynet is easily modulable and can be used for further studies such as adding new information (e.g. Sea Surface Temperature), or training with another ground truth.

# Some examples of the segmentation
![](example_eddynet.png)
<hr>


![](example_eddynet2.png)
<hr>


![](example_eddynet3.png)
