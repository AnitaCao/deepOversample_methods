import torchvision
from torchvision.models import resnet50
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
import os
import numpy as np
from torch import nn, optim
import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#print(os.getcwd())

#data path 
#deepsmotepth = '/Users/tcroot/Google Drive/My Drive/Research/DeepSMOTE' #local
deepsmotepth = '/home/tcvcs/DeepOversample_reference_methods/datasets_files/DeepSMOTE'

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
    def __init__(self, data, labels, transform=None):
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
    ToTensor(), 
    Normalize((0.5,), (0.5,)),
    Resize((224, 224)) # Resizing to fit ResNet50 input size
])


# define model for MNIST
def getModel():
  model = resnet50(num_classes = 10)
  model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  #optimizer = optim.Adadelta(model.parameters())
  optimizer = optim.Adam(model.parameters(), lr= 1e-3)
  loss_fn = nn.CrossEntropyLoss()
  return model.to(device), loss_fn, optimizer

def get_data_loader(trail_batch_size, val_batch_size):
  train_data = DeepSMOTE_MNIST(X_train, y_train, transform)
  test_data = DeepSMOTE_MNIST(X_test, y_test, transform)

  train_loader = DataLoader(train_data, batch_size=trail_batch_size, shuffle=True) # Since my total sample size is 4000*0.8, if i choose batch_size 64, i will have about 500 batches. That's why my dataloader will have 500 items(batches).
  test_loader = DataLoader(test_data, batch_size=val_batch_size)
  return train_loader, test_loader

#from sklearn.metrics import classification_report
def calculate_metric(metric_fn, true_y, pred_y):

  # multi class problems need to have averaging method
  # Use inspect to get the signature and arguments of the precision_score function
  metric_signature = inspect.signature(metric_fn)
  metric_args = list(metric_signature.parameters.keys())

  #if "average" in inspect.getfullargspec(metric_fn).args:
  if 'average' in metric_args:
    # print('here: ', inspect.getfullargspec(metric_fn)), this method getfullargspec is not the correct one, should use inspect.signature instead.
    return metric_fn(true_y, pred_y, average="weighted")
  else:
    return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, batch_size):
  # just an utility printing function
  for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
    print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def plot_learning_curve(losses):
  epochs = range(1, len(losses) + 1)

  # Plotting training and validation loss
  plt.figure(figsize=(12, 5))
  plt.plot(epochs, losses, label='Training Loss')
  #plt.plot(epochs, validation_losses, label='Validation Loss')
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
# model.eval()

train_dl, test_dl = get_data_loader(batch_size, batch_size)
batches = len(train_dl)
val_batches = len(test_dl)

losses = []

for epoch in range(epochs):
  total_loss = 0
  # progress bar
  progress = tqdm(enumerate(train_dl), desc='Loss: ', total=batches)
  #----------------------------TRAINING---------------------------------
  # set model to training
  model.train()

  for i, data in progress:

    X, y = data[0].to(device), data[1].to(device)
    #X = X.cuda()
    #y = y.cuda()

    #trainig step for single batch
    model.zero_grad() #clear the gradiant
    outputs = model(X) # pass input data to model and get the output.

    #y = y.to(torch.int64)
    output = loss_fn(outputs, y.to(torch.int64)) #calculate loss against the true labels.

    output.backward()
    optimizer.step()
    current_loss = output.item()
    total_loss += current_loss
    progress.set_description('Loss: {:.4f}'.format(total_loss/(i+1)))

  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  #------------------------------VALIDATION------------------------------
  val_losses = 0
  precision, recall, f1, accuracy = [], [], [], []
  model.eval()

  with torch.no_grad():
    for i, data in enumerate(test_dl):
      X, y = data[0].to(device), data[1].to(device)

      # the outputs of the model is a tentor with shape[64, 10], each of the image has 10 different probability indicating it's probability of being 10 different classes.
      outputs = model(X) # this get's the prediction from the network

      # out of all the 10 different probablities for each of the images, get the max probability one. That is the predicted class.
      predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction

      # Print the NumPy array
      # numpy_array = predicted_classes.cpu().numpy()
      # print(numpy_array)
      # print(collections.Counter(numpy_array))

      val_losses += loss_fn(outputs, y.to(torch.int64))

      # calculate P/R/F1/A metrics for batch
      for acc, metric in zip((precision, recall, f1, accuracy),(precision_score, recall_score, f1_score, accuracy_score)):
        acc.append(calculate_metric(metric, y.cpu(), predicted_classes.cpu()))

      #print(classification_report(y.cpu(), predicted_classes.cpu(), digits=4))

  print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
  print_scores(precision, recall, f1, accuracy, val_batches)

  losses.append(total_loss/batches) # for plotting learning curve

  #losses.append([(val_losses/val_batches), total_loss/batches])
  #print(validation_losses)

plot_learning_curve(losses)
print(f"Training time: {time.time()-start_ts}s")