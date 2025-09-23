import pickle
import os
import cv2
import numpy as np

# Define the path to the data directory
DATA_PATH = 'dataset/'  # Update this path as needed

# Load the test data from pickle files
with open(DATA_PATH + 'x_img_test.pkl', 'rb') as f:
    x_img_test = pickle.load(f)

with open(DATA_PATH + 'x_trac_test.pkl', 'rb') as f:
    x_trac_test = pickle.load(f)

with open(DATA_PATH + 'y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)


# Define the path to the output directory
OUTPUT_DIR = 'img/test-data/y/'

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Save the y_test images to the output directory
for i, img in enumerate(y_test):
    # Convert the image to a format suitable for saving (e.g., uint8)
    img_to_save = (img * 255).astype(np.uint8)
    
    # Define the output path for the image
    output_path = os.path.join(OUTPUT_DIR, f'y_test_{i}.png')
    
    # Save the image
    cv2.imwrite(output_path, img_to_save)

print(f"Saved {len(y_test)} images to {OUTPUT_DIR}")