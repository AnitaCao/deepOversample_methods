import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Download the Fashion MNIST dataset
train_data = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

import pandas as pd

# Function to convert dataset to DataFrame
def dataset_to_dataframe(dataset):
    data_list = []
    for image, label in dataset:
        # Convert image tensor to a flattened list
        image_flattened = image.numpy().flatten()
        # Insert the label at the beginning of the list
        data_list.append([label] + image_flattened.tolist())
    
    # Create a DataFrame
    columns = ['label'] + [f'pixel{i}' for i in range(1, 785)]
    df = pd.DataFrame(data_list, columns=columns)
    return df

# Convert training and testing sets to DataFrames
train_df = dataset_to_dataframe(train_data)
test_df = dataset_to_dataframe(test_data)

# Save the DataFrames to CSV files
train_df.to_csv('fashionMnist_train.csv', index=False)
test_df.to_csv('fashionMnist_test.csv', index=False)
