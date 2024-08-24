import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import os
import rasterio

# Define a custom dataset class for loading and preparing the hyperspectral data
class HyperspectralDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # Hyperspectral data patches
        self.labels = labels  # Corresponding ground truth labels
        self.transform = transform  # Any transformations to apply to the data

    def __len__(self):
        return len(self.data)  # Return the number of data points

    def __getitem__(self, idx):
        image = self.data[idx]  # Get the image patch at the specified index
        label = self.labels[idx]  # Get the corresponding label

        if self.transform:
            image = self.transform(image)  # Apply transformations if provided

        return image, label  # Return the image and label as a tuple

# Loading and normalizing the hyperspectral image and ground truth labels from .tif files
with rasterio.open('path/to/your/hyperspectral_image.tif') as src:
    data = src.read()  # Read the data (bands, height, width)

# Transpose to (height, width, bands)
data = np.transpose(data, (1, 2, 0))

# Normalize the hyperspectral image data between 0 and 1
data = (data - data.min()) / (data.max() - data.min())

# Load the ground truth labels from a .tif file
with rasterio.open('path/to/your/ground_truth.tif') as src_gt:
    gt = src_gt.read(1)  # Read the first band as ground truth

# Define the number of spectral bands in the hyperspectral image
n_channels = data.shape[2]  # Number of spectral bands in the hyperspectral image

# Prepare the data by extracting patches and their corresponding labels
def extract_patches(image, labels, patch_size=32, stride=16):
    patches = []
    patches_labels = []
    
    for i in range(0, image.shape[0] - patch_size + 1, stride):  # Loop over rows
        for j in range(0, image.shape[1] - patch_size + 1, stride):  # Loop over columns
            patch = image[i:i + patch_size, j:j + patch_size, :]  # Extract image patch
            patch_label = labels[i:i + patch_size, j:j + patch_size]  # Extract corresponding label patch
            patches.append(patch)  # Add image patch to the list
            patches_labels.append(patch_label)  # Add label patch to the list
    
    return np.array(patches), np.array(patches_labels)  # Return the extracted patches and labels

patch_size = 32  # Size of each image patch
stride = 16  # Stride for sliding window

# Extract patches and their corresponding labels from the entire dataset
patches, patches_labels = extract_patches(data, gt, patch_size, stride)

# Convert labels to categorical (one-hot encoding)
n_classes = len(np.unique(gt)) - 1  # Number of classes in the ground truth (excluding background if necessary)
patches_labels = patches_labels - 1  # Adjust labels to be zero-indexed (if necessary)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    patches, patches_labels, test_size=0.2, random_state=42, stratify=patches_labels.reshape(-1)
)

# Define the data loaders for training and validation
batch_size = 16 

# Create the t
