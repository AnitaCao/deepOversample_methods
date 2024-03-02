# Create_Imblanced_datasets.py

#This file is to create imblanced datasets and store them in txt files for DeepSMOTE (or other methods) to use.

#import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset,Dataset
import numpy as np
import os
#import zipfile
from PIL import Image
import os
import pandas as pd

# Desired class ratios
class_ratios = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
num_classes = 10

#--------------------------FMNIST-DATA-PROCESSING---------------------------------
def create_imblanced_fmnist(class_ratios, num_classes):
  transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
  train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform) 
  indices = [] # Create an empty list to store indices for each class
  # Generate indices for imbalanced dataset
  for i in range(num_classes):  # There are 10 classes in Fashion MNIST
    class_indices = np.where(train_data.targets == i)[0]
    imbalance_indices = np.random.choice(class_indices, class_ratios[i], replace=False)
    indices.extend(imbalance_indices) 
  imbalanced_dataset = Subset(train_data, indices) # Create the imbalanced dataset
  return imbalanced_dataset

# Function to save dataset
def save_fmnist_txt(image_file, label_file, imbalanced_dataset):
    with open(image_file, 'w') as img_file, open(label_file, 'w') as lbl_file:
        for idx in imbalanced_dataset.indices:
            image, label = imbalanced_dataset.dataset[idx] # Access the original dataset directly
            image = image * 0.5 + 0.5 # Undo normalization and convert to numpy
            image_array = image.numpy().squeeze()  # Remove channel dim and convert to numpy
            image_str = ' '.join(map(str, image_array.flatten()))  # Flatten and write the image data
            img_file.write(image_str + '\n')  # Write the image
            lbl_file.write(str(label) + '\n')  # Write the label

def load_fmnist_from_txt(image_file_path, label_file_path):
    # Read the image data
    with open(image_file_path, 'r') as img_file:
        images = [np.array(line.split(), dtype=float) for line in img_file]
    images = np.array(images)
    # Assuming the original shape was 1x28x28 (for Fashion MNIST), reshape accordingly
    images = images.reshape((-1, 1, 28, 28))  # -1 for automatic calculation of batch size
    # Read the label data
    with open(label_file_path, 'r') as lbl_file:
        labels = np.array([int(line.strip()) for line in lbl_file])
    return images, labels

#--------------------------CIFAR10-DATA-PROCESSING---------------------------------
def create_imblanced_cirfar10(class_ratios, num_classes):
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) 
  indices = []  
  # Generate indices for imbalanced dataset
  for class_id, num_samples in enumerate(class_ratios):
    class_indices = np.where(np.array(train_data.targets) == class_id)[0] # Find indices of all samples belonging to the current class
    # Randomly select 'num_samples' indices from the class_indices
    selected_indices = np.random.choice(class_indices, num_samples, replace=False)
    indices.extend(selected_indices)
  imbalanced_dataset = Subset(train_data, indices)  # Create the imbalanced dataset
  return imbalanced_dataset


def save_cifar10_txt(image_file, label_file, imbalanced_dataset):
    with open(image_file, 'w') as img_file, open(label_file, 'w') as lbl_file:
        for idx in imbalanced_dataset.indices:
            image, label = imbalanced_dataset.dataset[idx]
            # Normalize the image to [0, 1] and flatten
            image_array = image.numpy().flatten() * 0.5 + 0.5
            image_str = ' '.join(map(str, image_array))
            # Write the flattened image data and label to their respective files
            img_file.write(image_str + '\n')
            lbl_file.write(f"{label}\n")


def load_cifar10_from_txt(image_file_path, label_file_path):
    with open(image_file_path, 'r') as img_file:
        # Each line in the image file represents one image in flattened form
        images = [np.array(line.split(), dtype=float) for line in img_file]
    # Reshape the images to their original shape: (batch_size, 3, 32, 32)
    images = np.array(images).reshape(-1, 3, 32, 32)
    
    with open(label_file_path, 'r') as lbl_file:
        # Each line in the label file represents the label for one image
        labels = np.array([int(line.strip()) for line in lbl_file])    
    return images, labels

#--------------------------CELEBA-DATA-PROCESSING---------------------------------
# create a celebA dataset based on 5 classes of hair color: blonde, black, bald, brown and gray

# create a dataset class from folder

# Create a custom dataset class from the CelebA dataset
class CelebAHairColorDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels  # A dict mapping image filenames to hair color labels
        print("labels: ")
        print(labels)
        print(len(labels))
        self.transform = transform
        # Store the image filenames in a list
        self.image_filenames = list(labels.keys())
        print("list of image_filenames:")
        print(self.image_filenames)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        print("img_name:")
        print(img_name)
        label = self.labels[img_name]
        print("label:")
        print(label)
        image = Image.open(os.path.join(self.image_dir, img_name))
        if self.transform:
            image = self.transform(image)
        return image, label


def create_imblanced_celeba(image_dir,annotations_file):  
    # Define transformations
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    hair_color_classes = ['Blond_Hair', 'Black_Hair', 'Bald', 'Brown_Hair', 'Gray_Hair']
    class_ratios = [15000, 1500, 750, 300, 150]
    num_classes = 5
    annotation = pd.read_csv(annotations_file, delim_whitespace=True, skiprows=1)

    # Create the imbalanced dataset
    indices = []
    hair_color_labels = {}
    for class_id, num_samples in enumerate(class_ratios):
        # Find indices of all samples belonging to the current class
        class_indices = np.where(np.array(annotation[hair_color_classes[class_id]]) == 1)[0]
        print("class_indices: " )
        print(class_indices)
        # Randomly select 'num_samples' indices from the class_indices
        selected_indices = np.random.choice(class_indices, num_samples, replace=False)
        #select the images names from the image_dir based on selected indices and add label to the images and store image name and label pairs into a dictoinary
        for idx in selected_indices:
            img_name = annotation.index[idx]
            hair_color_labels[img_name] = class_id
    print("hair_color_labels: ")
    print(hair_color_labels)
    # Create the imbalanced dataset  
    imbalanced_dataset = CelebAHairColorDataset(image_dir, hair_color_labels, transform=transform)
    return imbalanced_dataset

# Save the imbalanced dataset to text files
def save_celeba_txt(image_file, label_file, imbalanced_dataset):
    with open(image_file, 'w') as img_file, open(label_file, 'w') as lbl_file:
        for idx in range(len(imbalanced_dataset)):
            image, label = imbalanced_dataset.__getitem__(idx)
            # Normalize the image to [0, 1] and flatten
            image_array = image.numpy().flatten()
            image_str = ' '.join(map(str, image_array))
            # Write the flattened image data and label to their respective files
            img_file.write(image_str + '\n')
            lbl_file.write(f"{label}\n")

def load_celeba_from_txt(image_file_path, label_file_path):
    with open(image_file_path, 'r') as img_file:
        # Each line in the image file represents one image in flattened form
        images = [np.array(line.split(), dtype=float) for line in img_file]
    # Reshape the images to their original shape: (batch_size, 3, 64, 64)
    images = np.array(images).reshape(-1, 3, 64, 64)
    
    with open(label_file_path, 'r') as lbl_file:
        # Each line in the label file represents the label for one image
        labels = np.array([int(line.strip()) for line in lbl_file])    
    return images, labels




#------------TESTING----------------------------------------

# File paths

parent_dir = os.path.dirname(os.getcwd())
deepsmotepth = os.path.join(parent_dir, 'datasets_files', 'DeepSMOTE')
#deepsmotepth = './datsets_files/DeepSMOTE' #pc local
#deepsmotepth = '/home/tcvcs/DeepOversample_reference_methods/datsets_files/DeepSMOTE' #server
imgtype = 'celebA'
image_0_set_path = os.path.join(deepsmotepth, imgtype, '0')
image_file_path = os.path.join(image_0_set_path, 'imbalanced_images.txt')
label_file_path = os.path.join(image_0_set_path, 'imbalanced_labels.txt')
#image_file_path = deepsmotepth + '/' + imgtype + '/0/' + 'imbalanced_images.txt'
#label_file_path = deepsmotepth + '/' + imgtype + '/0/' + 'imbalanced_labels.txt'

celeba_data_path = os.path.join(parent_dir, 'data', 'celeba')
image_dir = os.path.join(celeba_data_path, 'celeba_imgs')
attributes_path = os.path.join(celeba_data_path, 'list_attr_celeba.txt')





hair_color_classes = ['Blond_Hair', 'Black_Hair', 'Bald', 'Brown_Hair', 'Gray_Hair']
class_ratios = [15000, 1500, 750, 300, 150]
num_classes = 5
annotation = pd.read_csv(attributes_path, delim_whitespace=True, skiprows=1)

    # Create the imbalanced dataset
indices = []
hair_color_labels = {}
for class_id, num_samples in enumerate(class_ratios):
    # Find indices of all samples belonging to the current class
    class_indices = np.where(np.array(annotation[hair_color_classes[class_id]]) == 1)[0]
    print("class_indices: " )
    print(class_indices)
    # Randomly select 'num_samples' indices from the class_indices
    selected_indices = np.random.choice(class_indices, num_samples, replace=False)
    #select the images names from the image_dir based on selected indices and add label to the images and store image name and label pairs into a dictoinary
    for idx in selected_indices:
        img_name = annotation.index[idx]
        hair_color_labels[img_name] = class_id
print("hair_color_labels: ")
print(hair_color_labels)









imblanced_celeba = create_imblanced_celeba(image_dir,attributes_path)
print("Imbalanced dataset created successfully.")

save_celeba_txt(image_file_path, label_file_path, imblanced_celeba)
print("Dataset saved to text files successfully.")

# Load the datasets
images, labels = load_celeba_from_txt(image_file_path, label_file_path)
print("Dataset loaded successfully.")

# plot the images
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, ax in enumerate(axes):
    ax.imshow(images[i].transpose(1, 2, 0))
    ax.set_title(f"Label: {labels[i]}")
    ax.axis('off')  
plt.show()
#------------------------------------------------------------