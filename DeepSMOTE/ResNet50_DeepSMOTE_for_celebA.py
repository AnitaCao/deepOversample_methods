from torchvision.models import resnet50
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
import os
import numpy as np
from torch import nn, optim
import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale,ToPILImage
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#deepsmotepth = 'D:/anita/Research/DeepOversample_reference_methods/deepOversample_methods/datasets_files/DeepSMOTE' #local in pc.

# File paths
parent_dir = os.path.dirname(os.getcwd())
deepsmotepth = os.path.join(parent_dir, 'datasets_files', 'DeepSMOTE')

imgtype = 'celebA'
dtrnimg = deepsmotepth + '/' + imgtype + '/balanced_data/'

#5 classes, each class has 15000 samples.
tri_f = os.path.join(dtrnimg, "0_trn_img_b.txt")
print(tri_f)
trl_f = os.path.join(dtrnimg, "0_trn_lab_b.txt")
print(trl_f)


def load_data(image_path, lab_path):
  X_array = np.loadtxt(tri_f)
  y_array = np.loadtxt(trl_f)
  X = X_array.reshape(X_array.shape[0],3,64,64)
  y = y_array.astype(np.int32)
  return X, y

X, y = load_data(tri_f, trl_f)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#check the X_train shape
print(X_train.shape) # (60000, 3, 64, 64)
print(X_test.shape) # (15000, 3, 64, 64)
print(y_train.shape) # (60000,)
print(y_test.shape) # (15000,)
print(X_train[0].shape) # (3, 64, 64)

import collections
print(collections.Counter(y_test))
print(collections.Counter(y)) #the dataset should be balanced. 

'''
#----------------Display celebA data -----------------------

random_indices = np.random.choice(X.shape[0], 100, replace=False)

# Use the selected indices to extract the 100 data points
random_data = X[random_indices]
y_random = y[random_indices]
print(y_random)

for i in range(100):

    plt.subplot(10, 10, i+1)  # Create a 2x5 grid of subplots for 10 images
    plt.imshow(random_data[i].transpose((1,2,0)))  # Display the color image
    plt.axis('off')  # Turn off axis labels and ticks

plt.tight_layout()  # Ensure proper layout of subplots
plt.show()
#--------------------------------------------------------
'''
#define celebA dataset class based on X, y
class CelebADataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        if image.shape[0] == 3:
            image = image.transpose((1, 2, 0))
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the transformation for the data
transform = Compose([ToPILImage(), 
                     Resize((224,224)), 
                     ToTensor(), 
                     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])


# Create the model
def getModel():
    model = resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    return model, optimizer, loss_fn

def get_data_loader(train_batch_size, val_batch_size):
    # Create the dataset
    train_data = CelebADataset(X_train, y_train, transform=transform)
    test_data = CelebADataset(X_test, y_test, transform=transform)
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False)
    return train_loader, test_loader

#-----------------Define the metric calculation function---------------------
from sklearn.metrics import confusion_matrix, accuracy_score
  
def calculate_metric(true_y, pred_y):
    confMat=confusion_matrix(true_y, pred_y,labels=[0, 1, 2, 3, 4])
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


epochs = 20
batch_size = 64
start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")

model, loss_fn, optimizer = getModel()
print(model)
model.to(device)

train_dl, test_dl = get_data_loader(batch_size, batch_size)
tra_batches = len(train_dl)
val_batches = len(test_dl)

iter=int(epochs*len(train_dl)) #number of iterations, total number of batches
c=5  #number of classes

#initialize the variables to save the metrics
#I will save the metrics for each batch, so the size of the arrays will be iter.
#I only saved the testing metrics, but we can save the training metrics as well if needed.
acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))

train_losses, validation_losses = [], []
model = model.to(device)

best_val_loss = float('inf')

for epoch in range(epochs):
  model.train()
  train_loss = 0
  for i, (inputs, labels) in enumerate(train_dl):
    #X, y = data.to(device)
    X = inputs.to(device)
    y = labels.to(device)

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
    for i, (inputs, labels) in enumerate(test_dl):
      #X, y = data[0].to(device), data[1].to(device)
      X = inputs.to(device)
      y = labels.to(device)
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

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_model_path = deepsmotepth + '/' + imgtype + '/balanced_data/' + 'best_model.pth'
      torch.save(model.state_dict(), best_model_path)
      print(f'Best model saved at epoch {epoch+1}')

print(f"Training time: {time.time()-start_ts}s")

#-----------------Save the metrics---------------------
#save the metrics
savePath=deepsmotepth + '/' + imgtype + '/balanced_data/'
recordSave=savePath+'Record'
np.savez(recordSave, acsaSaveTr=acsaSaveTr, gmSaveTr=gmSaveTr, accSaveTr=accSaveTr, acsaSaveTs=acsaSaveTs, 
    gmSaveTs=gmSaveTs, accSaveTs=accSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs, tprSaveTr=tprSaveTr, tprSaveTs=tprSaveTs)

plot_learning_curve(train_losses,validation_losses)