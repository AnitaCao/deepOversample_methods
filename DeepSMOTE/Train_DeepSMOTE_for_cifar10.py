import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
from Create_Imblanced_datasets import *

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
"""set models, loss functions"""
# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


##############################################################################
"""functions to create SMOTE images"""

def biased_get_class(c):

    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]

    return xbeg, ybeg
    #return xclass, yclass

def G_SM(X, y,n_to_sample,cl):

    # determining the number of samples to generate
    #n_to_sample = 10

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

#xsamp, ysamp = SM(xclass,yclass)

###############################################################################
deepsmotepth = args['oversampler_path']
imgtype = args['dataset']
trnimgfile = deepsmotepth + '/' + imgtype + '/0/imbalanced_images.txt'
trnlabfile = deepsmotepth + '/' + imgtype + '/0/imbalanced_labels.txt'

encoder = Encoder(args)
print(encoder)

decoder = Decoder(args)
print(decoder)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
decoder = decoder.to(device)
encoder = encoder.to(device)

train_on_gpu = torch.cuda.is_available()

#decoder loss function
criterion = nn.MSELoss()
criterion = criterion.to(device)

print(trnimgfile)
print(trnlabfile)

dec_x, dec_y = load_cifar10_from_txt(trnimgfile,trnlabfile)

print('train imgs before reshape ',dec_x.shape)
print('train labels ',dec_y.shape)
print(collections.Counter(dec_y))


batch_size = 100
num_workers = 0

#torch.Tensor returns float so if want long then use torch.tensor
tensor_x = torch.Tensor(dec_x)
tensor_y = torch.tensor(dec_y,dtype=torch.long)
fmnist_dataset = TensorDataset(tensor_x,tensor_y)

train_loader = torch.utils.data.DataLoader(fmnist_dataset,
    batch_size=batch_size,shuffle=True,num_workers=num_workers)

classes = ('0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9')

best_loss = np.inf

t0 = time.time()
if args['train']:
    enc_optim = torch.optim.Adam(encoder.parameters(), lr = args['lr'])
    dec_optim = torch.optim.Adam(decoder.parameters(), lr = args['lr'])

    for epoch in range(args['epochs']):
        train_loss = 0.0
        tmse_loss = 0.0
        tdiscr_loss = 0.0
        # train for one epoch -- set nets to train mode
        encoder.train()
        decoder.train()

        for images,labs in train_loader:

            # zero gradients for each batch
            encoder.zero_grad()
            decoder.zero_grad()

            images, labs = images.to(device), labs.to(device)
            print(images.size())
            print(labs.size())

            labsn = labs.detach().cpu().numpy()
            print('labsn ',labsn.shape, labsn)

            # run images
            z_hat = encoder(images) 
            # print(z_hat.size()) #torch.Size([100, 300]), batch_size by feature_dimention

            x_hat = decoder(z_hat) #decoder outputs tanh
            #print('xhat ', x_hat.size())
            #print(x_hat)
            mse = criterion(x_hat,images)
            print('mse ',mse)

            resx = []
            resy = []

            tc = np.random.choice(10,1)
            #tc = 9
            xbeg = dec_x[dec_y == tc]

            ybeg = dec_y[dec_y == tc]


            xlen = len(xbeg)
            nsamp = min(xlen, 100)
            ind = np.random.choice(list(range(len(xbeg))),nsamp,replace=False)
            xclass = xbeg[ind]
            yclass = ybeg[ind]

            xclen = len(xclass)
            #print('xclen ',xclen)
            xcminus = np.arange(1,xclen)
            #print('minus ',xcminus.shape,xcminus)

            xcplus = np.append(xcminus,0)
            #print('xcplus ',xcplus)
            xcnew = (xclass[[xcplus],:])
            #xcnew = np.squeeze(xcnew)
            xcnew = xcnew.reshape(xcnew.shape[1],xcnew.shape[2],xcnew.shape[3],xcnew.shape[4])
            #print('xcnew ',xcnew.shape)

            xcnew = torch.Tensor(xcnew)
            xcnew = xcnew.to(device)

            #encode xclass to feature space
            xclass = torch.Tensor(xclass)
            xclass = xclass.to(device)
            xclass = encoder(xclass)
            #print('xclass ',xclass.shape)

            xclass = xclass.detach().cpu().numpy()

            xc_enc = (xclass[[xcplus],:])
            xc_enc = np.squeeze(xc_enc)
            #print('xc enc ',xc_enc.shape)

            xc_enc = torch.Tensor(xc_enc)
            xc_enc = xc_enc.to(device)

            ximg = decoder(xc_enc)

            mse2 = criterion(ximg,xcnew)

            comb_loss = mse2 + mse
            comb_loss.backward()

            enc_optim.step()
            dec_optim.step()

            train_loss += comb_loss.item()*images.size(0)
            tmse_loss += mse.item()*images.size(0)
            tdiscr_loss += mse2.item()*images.size(0)


        # print avg training statistics
        train_loss = train_loss/len(train_loader)
        tmse_loss = tmse_loss/len(train_loader)
        tdiscr_loss = tdiscr_loss/len(train_loader)
        print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,
                train_loss,tmse_loss,tdiscr_loss))


        #store the best encoder and decoder models
        #here, /crs5 is a reference to 5 way cross validation, but is not
        #necessary for illustration purposes
        if train_loss < best_loss:
            print('Saving..')
            path_enc = deepsmotepth + '/' + imgtype + '/0/bst_enc.pth'
            path_dec = deepsmotepth + '/' + imgtype + '/0/bst_dec.pth'

            torch.save(encoder.state_dict(), path_enc)
            torch.save(decoder.state_dict(), path_dec)

            best_loss = train_loss

    #in addition, store the final model (may not be the best) for
    #informational purposes
    path_enc = deepsmotepth + '/' + imgtype +  '/f_enc.pth'
    path_dec = deepsmotepth + '/' + imgtype +  '/f_dec.pth'

    torch.save(encoder.state_dict(), path_enc)
    torch.save(decoder.state_dict(), path_dec)
