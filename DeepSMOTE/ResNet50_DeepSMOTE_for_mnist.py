import torchvision
from torchvision.models import resnet50
from torchvision.datasets import MNIST
#from tqdm.autonotebook import tqdm
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
import os
import numpy as np
from torch import nn, optim
import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from torchsummary import summary
#print(os.getcwd())

#data path 
#deepsmotepth = '/Users/tcroot/Google Drive/My Drive/Research/DeepSMOTE' #local in macbook google drive
#deepsmotepth = '/home/tcvcs/DeepOversample_reference_methods/datasets_files/DeepSMOTE' #server
deepsmotepth = 'D:/anita/Research/DeepOversample_reference_methods/deepOversample_methods/datasets_files/DeepSMOTE' #local in pc.

imgtype = 'MNIST'
dtrnimg = deepsmotepth + '/' + imgtype + '/balanced_data/'

#10 classes, each class has 4000 samples, 32000 samples in total.
tri_f = os.path.join(dtrnimg, "0_trn_img_b.txt")
print(tri_f)
trl_f = os.path.join(dtrnimg, "0_trn_lab_b.txt")
print(trl_f)


def load_data(image_path, lab_path):
  X_array = np.loadtxt(tri_f)
  y_array = np.loadtxt(trl_f)
  X = X_array.reshape(X_array.shape[0],1,28,28)
  y = y_array.astype(np.int32)
  return X, y
  

X, y = load_data(tri_f, trl_f)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#check the X_train shape
print(X_train.shape)
print(X_test.shape) 
print(y_train.shape)
print(y_test.shape)
print(X_train[0].shape)

import collections
print(collections.Counter(y_test))
print(collections.Counter(y)) #the dataset should be balanced. 

'''
#----------------Display mnist data -----------------------

random_indices = np.random.choice(X.shape[0], 100, replace=False)

# Use the selected indices to extract the 100 data points
random_data = X[random_indices]
y_random = y[random_indices]
print(y_random)

for i in range(100):

    plt.subplot(10, 10, i+1)  # Create a 2x5 grid of subplots for 10 images
    plt.imshow(random_data[i, 0], cmap='gray')  # Display the grayscale image
    plt.axis('off')  # Turn off axis labels and ticks

plt.tight_layout()  # Ensure proper layout of subplots
plt.show()
#--------------------------------------------------------
'''


#define deepsmote balanced minist dataset
class DeepSMOTE_MNIST(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# Define the transformation for the data
transform = Compose([
    Resize((224,224)),  # Resize to 224x224 pixels
    #Grayscale(num_output_channels=3),  # Convert to 3 color channels
    ToTensor()  # Convert to tensor
    #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and standard deviation
])


# define model for MNIST
def getModel():
  model = resnet50(pretrained = True)
  model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 10) 
  optimizer = optim.Adam(model.parameters(), lr= 1e-3)
  loss_fn = nn.CrossEntropyLoss()
  return model.to(device), loss_fn, optimizer

def get_data_loader(trail_batch_size, val_batch_size):
  train_data = DeepSMOTE_MNIST(X_train, y_train, Compose([Resize((224,224)), ToTensor()])
  test_data = DeepSMOTE_MNIST(X_test, y_test, Compose([Resize((224,224)), ToTensor()])
  train_loader = DataLoader(train_data, batch_size=trail_batch_size, shuffle=True) # Since my total sample size is 4000*0.8, if i choose batch_size 64, i will have about 500 batches. That's why my dataloader will have 500 items(batches).
  test_loader = DataLoader(test_data, batch_size=val_batch_size)
  return train_loader, test_loader

#-----------------Define the metric calculation function---------------------
from sklearn.metrics import confusion_matrix, accuracy_score
  
def calculate_metric(true_y, pred_y):
    confMat=confusion_matrix(true_y, pred_y)
    nc=np.sum(confMat, axis=1)
    tp=np.diagonal(confMat)
    tpr=tp/nc
    acsa=np.mean(tpr)
    gm=np.prod(tpr)**(1/confMat.shape[0])
    acc=np.sum(tp)/np.sum(nc)
    return acsa, gm, tpr, confMat, acc

#-----------------Define the plot learning curve function---------------------
def plot_learning_curve(train_losses,validation_losses):
  # Plotting training and validation loss
  plt.figure(figsize=(12, 5))
  plt.plot(train_losses, label='Training Loss')
  plt.plot(validation_losses, label='Validation Loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.tight_layout()
  plt.show()


epochs = 10 #20
batch_size = 64
start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")

model, loss_fn, optimizer = getModel()
print(model)
#model.eval()

train_dl, test_dl = get_data_loader(batch_size, batch_size)
tra_batches = len(train_dl)
val_batches = len(test_dl)



iter=int(epochs*len(train_dl)) #number of iterations, total number of batches
c=10  #number of classes

acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))

train_losses, validation_losses = [], []

for epoch in range(epochs):
  model.train()
  train_loss = 0
  for i, data in enumerate(train_dl):
    X, y = data[0].to(device), data[1].to(device)
    model.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y.to(torch.int64))
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
      print(f'Epoch {epoch+1}/{epochs}, batch {i+1}/{tra_batches}, loss: {loss.item()}')
    train_loss += loss.item()
  
  train_losses.append(train_loss/tra_batches)
  print(f'Epoch {epoch+1}/{epochs}, training loss: {train_loss/tra_batches}')

  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  
  #-----------------Validation---------------------
  model.eval()
  val_loss = 0
  with torch.no_grad():
    for i, data in enumerate(test_dl):
      X, y = data[0].to(device), data[1].to(device)
      outputs = model(X)
      val_loss += loss_fn(outputs, y.to(torch.int64)).item()

      pLabels = torch.max(outputs, 1)[1] #get class from network's prediction
      acsa, gm, tpr, confMat, acc = calculate_metric(y.cpu(), pLabels.cpu())
      acsaSaveTs[epoch*val_batches+i], gmSaveTs[epoch*val_batches+i], accSaveTs[epoch*val_batches+i]=acsa, gm, acc
      confMatSaveTs[epoch*val_batches+i, :, :]=confMat
      tprSaveTs[epoch*val_batches+i, :]=tpr
      print(f'Epoch {epoch+1}/{epochs}, batch {i+1}/{val_batches}, acsa: {acsa}, gm: {gm}, acc: {acc}')
    validation_losses.append(val_loss/val_batches)
    print(f'Epoch {epoch+1}/{epochs}, validation loss: {val_loss/val_batches}')



print(f"Training time: {time.time()-start_ts}s")

plot_learning_curve(train_losses,validation_losses)


