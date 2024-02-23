# -*- coding: utf-8 -*-

import collections
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
print(torch.version.cuda) #10.1
import time
from Create_Imblanced_datasets import *

t0 = time.time()

args = {}
args['dim_h'] = 64         # factor controlling size of hidden layers
args['n_channel'] = 3 #1    # number of channels in the input data

args['n_z'] = 300 #600     # number of dimensions in latent space.

args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
args['epochs'] = 200        # 200 how many epochs to run for
args['batch_size'] = 100   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from

args['dataset'] = 'cifar10'  #'fmnist' # specify which dataset to use
args['oversampler_path'] = '/home/tcvcs/datasets_files/DeepSMOTE'

## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),

            #3d and 32 by 32
            #nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 1, 0, bias=False),

            nn.BatchNorm2d(self.dim_h * 8), # 40 X 8 = 320
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True) )#,
            #nn.Conv2d(self.dim_h * 8, 1, 2, 1, 0, bias=False))
            #nn.Conv2d(self.dim_h * 8, 1, 4, 1, 0, bias=False))
        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3)*2*2, self.n_z)


    def forward(self, x):
        #print('enc')
        print('input ',x.size()) #torch.Size([100, 3,32,32])
        x = self.conv(x)
        x = x.view(x.size(0), -1) ## Flatten the output of the conv layers to match the input shape of the fc layer
        
        x = self.fc(x)
        #print('out ',x.size()) #torch.Size([128, 20])
        #out  torch.Size([100, 300])
        return x
    
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 2 * 2),  # Adjusted for larger spatial size
            nn.ReLU()
        )

        # Deconvolutional filters, reversing the encoder's convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, 2, 1),
            nn.Tanh()  # Assuming the input was normalized to [-1, 1]
        )

    def forward(self, x):
        #print('dec')
        #print('input ',x.size())
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 2, 2)
        x = self.deconv(x)
        return x


##############################################################################
def biased_get_class1(c):

    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]

    return xbeg, ybeg
    #return xclass, yclass


def G_SM1(X, y,n_to_sample,cl):
    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample

############################################################################# 

#path on the computer where the best model is created
deepsmotepth = args['oversampler_path']
imgtype = args['dataset']
modpth = deepsmotepth + '/' + imgtype
encoder_path = os.path.join(modpth, "0/bst_enc.pth")
decoder_path = os.path.join(modpth, "0/bst_dec.pth")


trnimgfile = deepsmotepth + '/' + imgtype + '/0/imbalanced_images.txt'
trnlabfile = deepsmotepth + '/' + imgtype + '/0/imbalanced_labels.txt'

dec_x, dec_y = load_cifar10_from_txt(trnimgfile,trnlabfile)

print('train imgs shape ',dec_x.shape)
print('train labels shape ',dec_y.shape)
#print(collections.Counter(dec_y))

#store the counts of each class
class_counts = collections.Counter(dec_y)
print(class_counts)

#conver class_counts to a list
imbal_ratio = list(class_counts.values())
print(imbal_ratio)


#generate some images
train_on_gpu = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


encoder = Encoder(args)
encoder.load_state_dict(torch.load(encoder_path), strict=False)
encoder = encoder.to(device)

decoder = Decoder(args)
decoder.load_state_dict(torch.load(decoder_path), strict=False)
decoder = decoder.to(device)

encoder.eval()

#This part is only for testing
xbeg_ = dec_x[dec_y == 1]
ybeg_ = dec_y[dec_y == 1]
print(len(xbeg_)) #2000 imgs are 1
print(xbeg_.shape)   # for MNIST: (2000, 1, 28, 28)
xclass, yclass = biased_get_class1(1)
print(xclass.shape) # for MNIST: (2000, 1, 28, 28); BUT for cifar10(2000, 3, 32, 32)
print(yclass[0]) #(500,)

resx = []
resy = []

for i in range(1,10):
  xclass, yclass = biased_get_class1(i)

  #encode xclass to feature space
  xclass = torch.Tensor(xclass)
  xclass = xclass.to(device)
  xclass = encoder(xclass)
  print(xclass.shape) #torch.Size([500, 600])

  xclass = xclass.detach().cpu().numpy()
  n = imbal_ratio[0] - imbal_ratio[i]
  xsamp, ysamp = G_SM1(xclass,yclass,n,i)
  print(xsamp.shape) #(4500, 600)
  print(len(ysamp)) #4500
  ysamp = np.array(ysamp)
  print(ysamp.shape) #4500

  """to generate samples for resnet"""
  xsamp = torch.Tensor(xsamp)
  xsamp = xsamp.to(device)
  #xsamp = xsamp.view(xsamp.size()[0], xsamp.size()[1], 1, 1)
  #print(xsamp.size()) #torch.Size([10, 600, 1, 1])
  ximg = decoder(xsamp)

  ximn = ximg.detach().cpu().numpy()
  print(ximn.shape) #(4500, 3, 32, 32)
  #ximn = np.expand_dims(ximn,axis=1)
  print(ximn.shape) #(4500, 3, 32, 32)
  resx.append(ximn)
  resy.append(ysamp)
  #print('resx ',resx.shape)
  #print('resy ',resy.shape)
  #print()

resx1 = np.vstack(resx)
resy1 = np.hstack(resy)
print(resx1.shape) #(34720, 3, 32, 32)
#resx1 = np.squeeze(resx1)
print(resx1.shape) #(34720, 3, 32, 32)
print(resy1.shape) #(34720,)

resx1 = resx1.reshape(resx1.shape[0],-1)
print(resx1.shape) #(34720, 3072)

dec_x1 = dec_x.reshape(dec_x.shape[0],-1)
print('decx1 ',dec_x1.shape)
combx = np.vstack((resx1,dec_x1))
comby = np.hstack((resy1,dec_y))

print(combx.shape) #(45000, 3, 32, 32)
print(comby.shape) #(45000,)
print(collections.Counter(comby))

#-----------------testing the generated images-------------------
import matplotlib.pyplot as plt
random_indices = np.random.choice(dec_x.shape[0], 100, replace=False)

# Use the selected indices to extract the 100 data points
random_data = dec_x[random_indices]
y_random = dec_y[random_indices]
print(y_random)

for i in range(100):
    plt.subplot(10, 10, i+1)  # Create a 2x5 grid of subplots for 10 images
    plt.imshow(random_data[i, 0], cmap='gray')  # Display the grayscale image
    plt.axis('off')  # Turn off axis labels and ticks

plt.tight_layout()  # Ensure proper layout of subplots
plt.show()
#-----------------testing the generated images-------------------

#save the generated images
ifile = modpth + '/balanced_data/0_trn_img_b.txt'
np.savetxt(ifile, combx)

lfile = modpth + '/balanced_data/0_trn_lab_b.txt'
np.savetxt(lfile,comby)