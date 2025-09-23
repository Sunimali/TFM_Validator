import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2

#define the percentage of the maximum value
PERCENTAGE = 0.05

def save_img(data, path):
    output_image = np.clip(data, 0, 1)  # Clip values between 0 and 1

    # Save the image in grayscale and preserve dimensions
    plt.imsave(path, output_image, cmap='gray', format='png',)

# Function to add Gaussian noise
def add_noise(data, noise_level):
    # noise = np.random.normal(0, noise_level, data.shape)
    # noisy_data = data *(100 + noise)/100
    # return noisy_data  
    if isinstance(data, torch.Tensor):
        data = data.numpy() 
    #find the maximum value in the data
    max_val = np.max(data)

    #get the percentage of the maximum value
    max_val = max_val * PERCENTAGE

    #add noise to the data by random value and maximum value is defined
    data = data + np.random.normal(0, max_val * noise_level, data.shape)
    return data


def test_nosie_add(data):
    # Define the size of the corner where noise will be added
     #find maximum value in the data
    max_val = np.max(data)
    min_val = np.min(data)

    print(f"Max value: {max_val}")
    print(f"Min value: {min_val}")

    #add noise only for traction data  (x and y) [18:0,18:0] by random value and maximum value is defined
    data[0:18,0:18] = np.random.normal(min_val, max_val, data[0:18,0:18].shape)

    return data

    # Function to add Gaussian noise  

# Function to create a save path based on the noise level range and difference
def create_save_path(noise_levels, PLOTS_PATH):
    min_noise = noise_levels[0]
    max_noise = noise_levels[-1]
    noise_diff = noise_levels[1] - noise_levels[0]  # Calculate the difference between consecutive levels
    range_str = f"{min_noise:.3f}_{max_noise:.3f}_{noise_diff:.4f}"  # Create a string representation of the range and difference
    save_path = os.path.join(PLOTS_PATH, f'noise_range_{range_str}')  # Base path with noise range
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    return save_path   


# Function to calculate Dice coefficient
# def dice_coefficient(predicted, ground_truth, threshold=0.5):
#     predicted = predicted/255.0
#     ground_truth = ground_truth/255.0
#     predicted = (predicted > threshold).astype(float)
#     ground_truth = (ground_truth > 0.5).astype(float)

#     intersection = np.sum(predicted * ground_truth)
#     sum_predicted = np.sum(predicted)
#     sum_ground_truth = np.sum(ground_truth)

#     if sum_predicted + sum_ground_truth == 0:
#         return 1.0  # Both are empty, considered as perfect match

#     dice = (2.0 * intersection) / (sum_predicted + sum_ground_truth)

#     return intersection,sum_ground_truth,dice

def dice_coefficient(preds, targets, smooth=1e-6):
    # Threshold only for evaluation
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()  # Ensure binary targets
    intersection = (preds * targets).sum()
    denominator = preds.sum() + targets.sum()
    return (2 * intersection + smooth) / (denominator + smooth)

def add_gaussian_noise_to_cell(image, cell_mask, mean=0, std_dev=0.05):
    """
    Adds Gaussian noise within the cell boundary.

    Args:
    - image: Input traction image (numpy array).
    - cell_mask: Binary mask for cell boundary (1 inside, 0 outside).
    - mean: Mean of Gaussian noise.
    - std_dev: Standard deviation of Gaussian noise.

    Returns:
    - noisy_image: Image with added Gaussian noise inside the cell boundary.
    """
    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, std_dev, image.shape)
    # Apply the noise only inside the cell boundary
    noisy_image = image + gaussian_noise * cell_mask
    # Clip values to maintain the range
    noisy_image = np.clip(noisy_image, 0, 1)
    
    return noisy_image


def generate_boundary_mask(cell_mask, dilation_size=3):
    """
    Generates a boundary mask by dilating the cell mask.
    
    Args:
    - cell_mask: Binary mask of the cell boundary (1 inside, 0 outside).
    - dilation_size: Size for dilation to create boundary region.
    
    Returns:
    - boundary_mask: Binary mask representing the boundary region.
    """
    # Dilate the cell mask to create a thicker boundary
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_mask = cv2.dilate(cell_mask, kernel, iterations=1)
    
    # The boundary mask is the difference between the dilated mask and the original mask
    boundary_mask = dilated_mask - cell_mask
    
    return boundary_mask

def add_gaussian_noise_to_boundary_xy(traction_image_x, traction_image_y, boundary_mask, mean=0, std_dev=0.05):
    """
    Adds Gaussian noise to the x and y components within the boundary region.
    
    Args:
    - traction_image_x: Traction data for the x component (numpy array).
    - traction_image_y: Traction data for the y component (numpy array).
    - boundary_mask: Binary mask for boundary region (1 in boundary, 0 elsewhere).
    - mean: Mean of Gaussian noise.
    - std_dev: Standard deviation of Gaussian noise.
    
    Returns:
    - noisy_traction_image_x: Noisy x component.
    - noisy_traction_image_y: Noisy y component.
    """
    # Generate Gaussian noise for x and y components
    gaussian_noise_x = np.random.normal(mean, std_dev, traction_image_x.shape)
    gaussian_noise_y = np.random.normal(mean, std_dev, traction_image_y.shape)
    
    # Apply the noise only within the boundary region
    noisy_traction_image_x = traction_image_x + gaussian_noise_x * boundary_mask
    noisy_traction_image_y = traction_image_y + gaussian_noise_y * boundary_mask
    
    # Clip values to maintain the range [0, 1] (or appropriate range for your data)
    noisy_traction_image_x = np.clip(noisy_traction_image_x, 0, 1)
    noisy_traction_image_y = np.clip(noisy_traction_image_y, 0, 1)
    
    return noisy_traction_image_x, noisy_traction_image_y
