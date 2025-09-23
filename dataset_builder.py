from matlab_adapter import mat_to_np
import os
import numpy as np
from PIL import Image
import pickle
import cv2
from scipy.io import loadmat

# Configuration
# PATH_PREFIX = '/home/local2/sunimali/research/TFM_prediction/Trial_4/'
PATH_PREFIX = '/home/local2/FTTC/Code/Trial_4/'
CELL_IDS = [2, 3, 5, 6, 11, 12, 14, 15, 16]
DESTINATION = 'dataset/new/'
TRAC_SCALAR = 1e+6  # Adjusted from 1e+12

# Constants
IMG_SIZE = 288
TRAC_SHAPE = (2, 36, 36)

def rearrange_blocks(traction):
    """
    Rearranges the blocks of the grid from the order (1, 2, 3, 4) to (4, 3, 2, 1).
    This function swaps the quadrants in reverse order.

    Parameters:
    traction (np.ndarray): Traction field of shape 2, H, W), where H and W are even.

    Returns:
    np.ndarray: Rearranged traction field.
    """

    _, height, width = traction.shape

    # Split the traction grid into 4 quadrants (2D slices along the first two axes)
    top_left = traction[:,:height//2, :width//2]       # Block 1
    top_right = traction[:,:height//2, width//2:]      # Block 2
    bottom_left = traction[:,height//2:, :width//2]    # Block 3
    bottom_right = traction[:,height//2:, width//2:]   # Block 4

    # Rearrange blocks in the new order: 4, 3, 2, 1
    rearranged = np.zeros_like(traction)
    rearranged[:,:height//2, :width//2] = bottom_right  # Block 4 to top-left
    rearranged[:,:height//2, width//2:] = bottom_left   # Block 3 to top-right
    rearranged[:,height//2:, :width//2] = top_right     # Block 2 to bottom-left
    rearranged[:,height//2:, width//2:] = top_left      # Block 1 to bottom-right

    return rearranged

def boundary_to_mask(boundary):
    """Convert MATLAB boundary to binary mask"""
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    boundary = boundary.astype(np.int32)
    
    boundary = boundary.astype(np.int32)
    
    # 2. Verify coordinate format
    if boundary.shape[0] == 2:  # If MATLAB stores as (2, N) array
        boundary = boundary.T   # Convert to (N, 2)
    
    # 3. Reshape for OpenCV compatibility
    boundary = boundary.reshape((-1, 1, 2))
    
    # 4. Handle empty/invalid boundaries
    if len(boundary) < 3:  # Minimum 3 points for a polygon
        return mask
        
    cv2.fillPoly(mask, [boundary], color=255)
    return mask
    # # Close the contour if needed
    # if not np.array_equal(boundary[0], boundary[-1]):
    #     boundary = np.vstack([boundary, boundary[0]])
        
    # cv2.fillPoly(mask, [boundary], color=255)
    # return mask.astype(np.float32) / 255.0

def process_image(img_path):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('L')  # Force grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0

def validate_data(cell_dir):
    """Check all required files exist"""
    required_files = [
        os.path.join(cell_dir, 'cropCell200001.bmp.tif'),
        os.path.join(cell_dir, 'cropCell200002.bmp.tif'),
        os.path.join(cell_dir, 'traction.csv'),
        os.path.join(cell_dir, 'tractionDLTFMs.csv'),
        os.path.join(cell_dir, 'Cellboundary1.mat')
    ]
    return all(os.path.exists(f) for f in required_files)

if __name__ == '__main__':    
    x, y,x_dl = [], [],[]
    skipped = 0

    for cell_id in CELL_IDS:
        for frame_id in range(300):
            cell_dir = os.path.join(
                PATH_PREFIX, 
                f'Cell_{cell_id}', 
                f'Cell{frame_id + 1}'
            )
            
            if not validate_data(cell_dir):
                skipped += 1
                continue

            # Process boundary
            boundary = mat_to_np(os.path.join(cell_dir, 'Cellboundary1.mat'))
            mask = boundary_to_mask(boundary)

            # Process images
            ref_img = process_image(os.path.join(cell_dir, 'cropCell200001.bmp.tif'))
            def_img = process_image(os.path.join(cell_dir, 'cropCell200002.bmp.tif'))

            # Process traction
            trac_data = np.genfromtxt(
                os.path.join(cell_dir, 'traction.csv'), 
                delimiter=','
            ).reshape(TRAC_SHAPE) * TRAC_SCALAR


            # Rearrange traction blocks
            trac_data = rearrange_blocks(trac_data)

            trac_data_x = trac_data[0].T
            trac_data_y = trac_data[1].T

            trac_data = np.array([trac_data_x, trac_data_y])

            #process tractionDLTFMs
            dl_trac_data = np.genfromtxt(
                os.path.join(cell_dir, 'tractionDLTFMs.csv'), 
                delimiter=','
            ).reshape(TRAC_SHAPE) * TRAC_SCALAR

            dl_trac_data = rearrange_blocks(dl_trac_data)

            dl_trac_data_x = dl_trac_data[0].T
            dl_trac_data_y = dl_trac_data[1].T

            dl_trac_data = np.array([dl_trac_data_x, dl_trac_data_y])

            x_dl.append([
                np.stack([ref_img, def_img], axis=0),  # Shape (2, 288, 288)
                dl_trac_data
            ])

            x.append([
                np.stack([ref_img, def_img], axis=0),  # Shape (2, 288, 288)
                trac_data
            ])
            y.append(mask)

    print(f"Processed {len(x)} samples, skipped {skipped} invalid entries")
    
    # Save data
    os.makedirs(DESTINATION, exist_ok=True)
    with open(os.path.join(DESTINATION, 'x.pkl'), 'wb') as f:
        pickle.dump(x, f)
    with open(os.path.join(DESTINATION, 'y.pkl'), 'wb') as f:
        pickle.dump(y, f)
    with open(os.path.join(DESTINATION, 'x_dl.pkl'), 'wb') as f:
        pickle.dump(x_dl, f)



