import torchvision
from torchvision.models import resnet18
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


deepsmotepth = '/home/tcvcs/DeepOversample_reference_methods/datasets_files/DeepSMOTE'

imgtype = 'cifar10'
dtrnimg = deepsmotepth + '/' + imgtype + '/balanced_data/'

tri_f = os.path.join(dtrnimg, "0_trn_img_b.txt")
print(tri_f)
trl_f = os.path.join(dtrnimg, "0_trn_lab_b.txt")
print(trl_f)

def load_data(image_path, lab_path):
  X_array = np.loadtxt(tri_f)
  y_array = np.loadtxt(trl_f)
  X = X_array.reshape(X_array.shape[0],3,32,32)
  y = y_array.astype(np.int32)
  return X, y
  

X, y = load_data(tri_f, trl_f)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#----------------Display cifar10 data -----------------------
'''
import collections
print(collections.Counter(y_test))
print(collections.Counter(y)) #the dataset should be balanced. 

random_indices = np.random.choice(X.shape[0], 10, replace=False)

# Use the selected indices to extract the 100 data points
random_data = X[random_indices]
y_random = y[random_indices]
print(y_random)

for i in range(10):

    plt.subplot(2, 5, i+1)  # Create a 2x5 grid of subplots for 10 images
    plt.imshow(random_data[i, 0], cmap='gray')  # Display the grayscale image
    plt.axis('off')  # Turn off axis labels and ticks

plt.tight_layout()  # Ensure proper layout of subplots
plt.show()
'''
#--------------------------------------------------------

#define cifar10 dataset class based on X, y
class Cifar10Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label 
    
    
# Define the transformation for the data
#transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create the dataset
train_dataset = Cifar10Dataset(X_train, y_train, transform=torch.Tensor)
test_dataset = Cifar10Dataset(X_test, y_test, transform=torch.Tensor)
# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Create the model
model = resnet18(num_classes = 10)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Move the model to the device
model.to(device)

# Define the evaluation metric
def metric(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

#---------------------------------------------------
'''
# Define the training function
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        print(outputs.shape)
        print(targets.shape)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return model, optimizer, epoch_loss
'''

# Define the evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_predictions = []
    running_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluation'):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.long()  # Convert targets to Long data type
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, 1)
            running_predictions.extend(predictions.cpu().numpy())
            running_targets.extend(targets.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    precision, recall, f1, accuracy = metric(running_targets, running_predictions)
    return epoch_loss, precision, recall, f1, accuracy

# Train the model
epochs = 10
train_losses = []
test_losses = []
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []

best_model_pth = deepsmotepth + '/' + imgtype + '/balanced_data/best_resnetmodel.pth'

start = time.time()
best_loss = float('inf')

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')

    #model, optimizer, train_loss = train(model, train_loader, criterion, optimizer, device)

    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.to(torch.long)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    #save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_pth)

    test_loss, precision, recall, f1, accuracy = evaluate(model, test_loader, criterion, device)
    train_losses.append(epoch_loss)
    test_losses.append(test_loss)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    accuracy_scores.append(accuracy)
    print(f'Training Loss: {epoch_loss:.4f}')
    print(f'Validation Loss: {test_loss:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print('-' * 10)

print(f'Training time: {time.time()-start:.0f} seconds')


# Plot the training and validation loss
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), test_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Plot the precision, recall, f1 and accuracy
plt.plot(range(1, epochs+1), precision_scores, label='Precision')
plt.plot(range(1, epochs+1), recall_scores, label='Recall')
plt.plot(range(1, epochs+1), f1_scores, label='F1')
plt.plot(range(1, epochs+1), accuracy_scores, label='Accuracy')
plt.title('Evaluation Metrics')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.show()


'''
# Save the model
torch.save(model.state_dict(), 'resnet18_cifar10.pt')

# Load the model
model = resnet18(num_classes = 10)
model.load_state_dict(torch.load('resnet18_cifar10.pt'))
model.to(device)

# Evaluate the model
test_loss, precision, recall, f1, accuracy = evaluate(model, test_loader, criterion, device)
print(f'Validation Loss: {test_loss:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# Make predictions
model.eval()
inputs, targets = next(iter(test_loader))
inputs, targets = inputs.to(device), targets.to(device)
outputs = model(inputs)
predictions = torch.argmax(outputs, 1)
print('Predicted:', predictions)
print('Ground Truth:', targets)

# Display the images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(inputs[i, 0].cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f'Predicted: {predictions[i].item()}, Target: {targets[i].item()}')
plt.tight_layout()
plt.show()

'''











