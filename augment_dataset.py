import PIL
import torch 
import pickle
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torchvision.transforms as T
import os
import torch 
import pickle
from PIL import Image
import numpy as np
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

DATA_PATH = 'dataset/new/'

class AddGaussianNoise(object):
    def __init__(self, mean=0., std_range=(0.01, 0.1)):
        self.mean = mean
        self.std_range = std_range
        
    def __call__(self, tensor):
        std = np.random.uniform(*self.std_range)
        return tensor + torch.randn(tensor.shape) * std + self.mean


def horizontal_flip(img1, img2, x_tract, y):

    #convert to tensor
    img1 = torch.tensor(img1)
    img2 = torch.tensor(img2)
    y = torch.tensor(y)
    x_tract = torch.tensor(x_tract)

    img1_copy = img1.clone()
    img2_copy = img2.clone()
    y_copy = y.clone()
    x_tract_copy = x_tract.clone()

    img1_copy = torch.flip(img1_copy, [1])
    img2_copy = torch.flip(img2_copy, [1])
    y_copy = torch.flip(y_copy, [1])

    x_tract_copy_0 = torch.flip(x_tract_copy[0], [1])*-1
    x_tract_copy_1 = torch.flip(x_tract_copy[1], [1])

    #convert back to numpy
    img1_copy = img1_copy.numpy()
    img2_copy = img2_copy.numpy()
    y_copy = y_copy.numpy()
    x_tract_copy = np.array((x_tract_copy_0.numpy(), x_tract_copy_1.numpy()))

    return img1_copy, img2_copy, x_tract_copy, y_copy

def vertical_flip(img1, img2, x_tract, y):
    #convert to tensor
    img1 = torch.tensor(img1)
    img2 = torch.tensor(img2)
    y = torch.tensor(y)
    x_tract = torch.tensor(x_tract)

    img1_copy = img1.clone()
    img2_copy = img2.clone()
    y_copy = y.clone()
    x_tract_copy = x_tract.clone()

    img1_copy = torch.flip(img1_copy, [0])
    img2_copy = torch.flip(img2_copy, [0])
    y_copy = torch.flip(y_copy, [1])
    x_tract_copy_0 = torch.flip(x_tract_copy[0], [0])
    x_tract_copy_1 = torch.flip(x_tract_copy[1], [0])*-1
    
    #convert back to numpy
    img1_copy = img1_copy.numpy()
    img2_copy = img2_copy.numpy()
    y_copy = y_copy.numpy()
    x_tract_copy = np.array((x_tract_copy_0.numpy(), x_tract_copy_1.numpy()))

    return img1_copy, img2_copy, x_tract_copy, y_copy

def rotate_90(img1, img2, x_tract, y):
    #convert to tensor
    img1 = torch.tensor(img1)
    img2 = torch.tensor(img2)
    y = torch.tensor(y)
    x_tract = torch.tensor(x_tract)

    img1_copy = img1.clone()
    img2_copy = img2.clone()
    y_copy = y.clone()
    x_tract_copy = x_tract.clone()

    img1_copy = torch.rot90(img1_copy, 1, [0, 1])
    img2_copy = torch.rot90(img2_copy, 1, [0, 1])
    y_copy = torch.rot90(y_copy, 1, [0, 1])
    x_tract_copy_0 = torch.rot90(x_tract_copy[1], 1, [0, 1])
    x_tract_copy_1 = torch.rot90(x_tract_copy[0], 1, [0,1])*-1
    
    #convert back to numpy
    img1_copy = img1_copy.numpy()
    img2_copy = img2_copy.numpy()
    y_copy = y_copy.numpy()
    x_tract_copy = np.array((x_tract_copy_0.numpy(), x_tract_copy_1.numpy()))


    return img1_copy, img2_copy, x_tract_copy, y_copy

def split_dataset(x_img, x_tract, y, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, random_seed=42):
    """
    Splits the dataset into train, validation, and test sets while maintaining alignment between X and y.

    Args:
        X (np.array): The input features (traction data).
        y (np.array): The corresponding labels (cell data).
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Splits must sum to 1"

    # First split: train and temp (val + test)
    x_img_train, x_img_temp,  x_tract_train, x_tract_temp, y_train, y_temp = train_test_split(x_img, x_tract, y, test_size=(val_ratio + test_ratio), random_state=random_seed)

    # Second split: validation and test
    x_img_val,  x_img_test, x_tract_val, x_tract_test, y_val, y_test = train_test_split(x_img_temp, x_tract_temp, y_temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed)

    return x_img_train, x_img_val, x_img_test, x_tract_train, x_tract_val, x_tract_test, y_train, y_val, y_test
def process_dataset(x_img_train, x_trac_train, y_train):
    # Split the dataset
    # x_img_train, x_img_val, x_img_test, x_trac_train, x_trac_val, x_trac_test, y_train, y_val, y_test = split_dataset(x_img, x_trac, y)
    # 
    x_train_aug, y_train_aug = [], []
    
    # Process training data with augmentation
    for i in range(len(x_img_train)):
        img1, img2 = x_img_train[i][0], x_img_train[i][1]

        
        x_aug_1, x_aug_2, x_trac_aug, y_aug = horizontal_flip(img1, img2, x_trac_train[i], y_train[i])
        
        # Augmented pair
        x_train_aug.append((np.stack((x_aug_1, x_aug_2)), x_trac_aug))
        y_train_aug.append(y_aug)
        # Vertical flip
        x_aug_1, x_aug_2, x_trac_aug, y_aug = vertical_flip(img1, img2, x_trac_train[i], y_train[i])
        x_train_aug.append((np.stack((x_aug_1, x_aug_2)), x_trac_aug))
        y_train_aug.append(y_aug)
        # Rotate 90 degrees
        x_aug_1, x_aug_2, x_trac_aug, y_aug = rotate_90(img1, img2, x_trac_train[i], y_train[i])
        x_train_aug.append((np.stack((x_aug_1, x_aug_2)), x_trac_aug))
        y_train_aug.append(y_aug)

    return x_train_aug, y_train_aug
    

if __name__ == '__main__':
    with open(DATA_PATH + 'x.pkl', 'rb') as f:
        x = pickle.load(f)
    with open(DATA_PATH + 'y.pkl', 'rb') as f:
        y = np.array(pickle.load(f))


    with open(DATA_PATH + 'train_p3_correct_x.pkl', 'rb') as f:
        train_x_correct = pickle.load(f)
    with open(DATA_PATH + 'train_p3_correct_y.pkl', 'rb') as f:
        train_y_correct = pickle.load(f)
    with open(DATA_PATH + 'train_p3_x.pkl', 'rb') as f:  
        train_x_de = pickle.load(f)
    with open(DATA_PATH + 'train_p3_y.pkl', 'rb') as f:
        train_y_de = pickle.load(f)
    with open(DATA_PATH + 'val_p3_x.pkl', 'rb') as f:
        val_x_de = pickle.load(f)
    with open(DATA_PATH + 'val_p3_y.pkl', 'rb') as f:
        val_y_de = pickle.load(f)


    x_img_train, x_trac_train = [], []
    print("x shape:", len(x))
    for x_i in train_x_correct:
        x_img_train.append(x_i[0])
        x_trac_train.append(x_i[1])

    

    print("x_img_train shape:", len(x_img_train))
    print("x_trac_train shape:", len(x_trac_train))
    print("xtrac shape 0:", x_trac_train[0].shape)
    print("xtrac shape 1:", x_trac_train[1].shape)
    print("xtrac shape 2:", x_trac_train[2].shape)
    print("xtrac shape 3:", x_trac_train[3].shape)

    x_train_aug, y_train_aug = process_dataset(x_img_train, x_trac_train, y)


    print("Training data length:", len(x_train_aug)) #correct

    print("training data length dilate & eroed:", len(train_x_de)) #dilate & erode

    # Verify shapes
    print("Training data shape:", x_train_aug[0][0].shape)  # Should be (2, H, W)
    print("Training data shape:", x_train_aug[0][1].shape)  # Should be (2, H, W)
    


    #verfiy dialte & erode shapes
    print("Training data shape dilate & erode:", train_x_de[1][0].shape)  # Should be (2, H, W)
    print("Training data shape dilate & erode:", train_x_de[1][1].shape)  # Should be (2, H, W)
    print("Validation data shape dilate & erode:", val_x_de[1][0].shape)  # Should be (2, H, W)
    print("Validation data shape dilate & erode:", val_x_de[1][1].shape)  # Should be (2, H, W)

    x_train_final = []
    for i in range(len(x_train_aug)):
        x_train_final.append(x_train_aug[i])
    for i in range(len(train_x_de)):
        x_train_final.append(train_x_de[i])


    y_train_final = np.concatenate((y_train_aug, train_y_de), axis=0)

    print("y_train_final shape:", y_train_final.shape)
   
    print("Training data length:", len(x_train_final)) #correct
    print("Validation data length:", len(val_x_de)) #correct

    # Save processed data
    with open(DATA_PATH + 'x_new_p3.pkl', 'wb') as f:
        pickle.dump({'train': x_train_final, 'test': val_x_de}, f)

    with open(DATA_PATH + 'y_new_p3.pkl', 'wb') as f:
        pickle.dump({'train': y_train_final, 'test': val_y_de}, f)
