#Evaulate the model on dialated boundaries (NM1)
import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from networks import UNET
import matplotlib.pyplot as plt
from PIL import Image
from utils import create_save_path, dice_coefficient,save_img

import cv2
def add_gaussian_for_dialation(traction, mask,std=0.001, mean=0, kernel_size=2):

    # Dilate the mask to create a border area around high values
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)  # Expands the mask

    # Create the noise application region (dilated mask minus original mask)
    border_region = dilated_mask - mask  

    # Generate Gaussian noise
    noise = np.random.normal(mean, std, traction.shape)
    # Apply noise only in the border region
    traction_noisy = traction.copy()
    traction_noisy[:,border_region == 255] += noise[:,border_region == 255]  # Add noise to the border

    return traction_noisy
def add_gaussian_for_dialation_cell(mask,std=0.001, mean=0, kernel_size=2):

    # Dilate the mask to create a border area around high values
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=3)  # Expands the mask

    return dilated_mask
dialated_kernel_sizes = [2, 3,4,5,6,7]

x_test_dialated_per_kernel = []
y_test_dialated_per_kernel = []
for kernel_size in dialated_kernel_sizes:


    # CONSTANTS
    # MODEL_PATH = '../weights/mid_inj_81.pth'
    # MODEL_PATH = '../weights/new/final_model.pth'  # Path to the trained model
    # MODEL_PATH = '../weights/new/best_model.pth'  # Path to the trained model
    MODEL_PATH = '../weights/new/de_best_model.pth'  # Path to the trained model
    # MODEL_PATH = '../weights/new/p3_best_model.pth'  # Path to the trained model
    OUTPUT_BASE_PATH = '../img/validation/new/phase2/dialation/'  # Base directory to save output images
    # CELL_IDS = [2, 3, 6 ,11,15,16] # Update with other cell IDs as needed
    CELL_IDS = [23] # Update with other cell IDs as needed
    # Define the path to the data directory
    DATA_PATH = '../dataset/new/'  # Update this path as needed

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_device = torch.device(device)

    # Load the test data from pickle files
    with open(DATA_PATH + 'x_test_correct.pkl', 'rb') as f:
        x_test = pickle.load(f)

    # with open(DATA_PATH + 'x_tract_per_cell.pkl', 'rb') as f:
    #     x_trac_test = pickle.load(f)

    with open(DATA_PATH + 'y_test_correct.pkl', 'rb') as f:
        y_test = pickle.load(f)

    # Convert lists of numpy arrays to numpy arrays for easier indexing
    #first element of tuple is x_img_test, second element tuple is x_trac_test and x_test is list of tuples
    x_img_test = np.array([x[0] for x in x_test])
    x_trac_test = np.array([x[1] for x in x_test])
    y_test = np.array(y_test)
    x_test_dialated = []
    

    print(f"y_test shape: {y_test.shape}")
    print(f"x_img_test shape: {x_img_test.shape}")
    print(f"x_trac_test shape: {x_trac_test.shape}")

    #create the dialated mask
    y_test_dialated = np.zeros_like(y_test)
    for i in range(y_test.shape[0]):
        y_test_dialated[i] = add_gaussian_for_dialation_cell(y_test[i], kernel_size = kernel_size)

    #create dialated tractions
    x_trac_test_dialated = np.zeros_like(x_trac_test)
    for i in range(x_trac_test.shape[0]):
        #calculate correct traction magnitude
        x_trac_mask = np.zeros((36, 36))
        x_trac_magnitude = np.sqrt(np.sum(x_trac_test[i]**2, axis=0))
        # print(f"x_trac_magnitude shape: {x_trac_magnitude.shape}")
        # print(f"x_trac_mask shape: {x_trac_mask.shape}")
        x_trac_magnitude = x_trac_magnitude / np.max(x_trac_magnitude)
        x_trac_magnitude = x_trac_magnitude * 255
        x_trac_magnitude = x_trac_magnitude.astype(np.uint8)
        x_trac_mask[x_trac_magnitude > 5] = 255
        x_trac_mask = x_trac_mask.astype(np.uint8)

        #add noise to the traction
        x_trac_test_dialated[i] = add_gaussian_for_dialation(x_trac_test[i], x_trac_mask, kernel_size=kernel_size)
        x_test_dialated.append((x_img_test[i], x_trac_test_dialated[i]))

    
    x_test_dialated_per_kernel.append(x_test_dialated)
    y_test_dialated_per_kernel.append(y_test_dialated)


    # Initialize a dictionary to store mean Dice scores by cell ID for comparison
    dice_score_by_cell = {}

    # Loop over each cell ID and process its data separately
    for index, cell_id in enumerate(CELL_IDS):
        print(f"\nProcessing cell ID: {cell_id}")

        # Ensure each cell ID pulls unique data from the dataset
        x_img_cell = torch.from_numpy(x_img_test).float()
        x_trac_cell = torch.from_numpy(x_trac_test_dialated).float()
        #y_cell need to divide by 255.0
        y_cell = torch.from_numpy(y_test/255.0).float()

        # Reload the model for each cell to prevent potential state carryover
        model = UNET()
        model = nn.DataParallel(model).to(torch_device)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        # Normalize data (done per cell to ensure each set is normalized independently)
        x_img_cell = (x_img_cell - x_img_cell.mean()) / x_img_cell.std()
        x_trac_cell = (x_trac_cell - x_trac_cell.mean()) / x_trac_cell.std()

        # y_cell = (y_cell - y_cell.mean()) / y_cell.std()

        # Set noise levels (assuming no noise for now)
        noise_levels = {kernel_size}

        # Create output directory for the current cell ID
        cell_dir = os.path.join(OUTPUT_BASE_PATH, f'cell{cell_id}')
        os.makedirs(cell_dir, exist_ok=True)

        # Process data for each noise level
        for noise_level in noise_levels:
            # Create directory for the current noise level in the cell directory
            noise_dir = os.path.join(cell_dir, f'noise{noise_level}')
            os.makedirs(noise_dir, exist_ok=True)

            # Add noise to x_trac_cell if needed
            x_trac_test_noisy = x_trac_cell  # No noise here, just placeholder

            # Prepare for batched processing
            batch_size = 10
            test_batches = len(x_img_cell) // batch_size
            dice_scores = []
            unique_outputs = []

            with torch.no_grad():
                for i in range(test_batches):
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size

                    # Prepare batch data
                    test_data_img = x_img_cell[start_idx:end_idx].to(torch_device).float()
                    test_data_trac = x_trac_test_noisy[start_idx:end_idx].to(torch_device).float()
                    test_y_batch = y_cell[start_idx:end_idx].to(torch_device).float()

                    # Model prediction
                    outputs = model((test_data_img, test_data_trac))
                    outputs_np = outputs.detach().cpu().numpy()  # Move to CPU and convert to numpy array

                    # # save_img(outputs[0].cpu().numpy(), f'img/epoch_{epoch+1}_result.png')
                    y_test_batch_np = test_y_batch.cpu().numpy()

                    # Save output images and compute Dice scores
                    for j, output in enumerate(outputs_np):
                        img_out = output
                        img_gt = y_test_batch_np[j]

                        #get the dice score using sigmoid of output
                        dice_scores.append(dice_coefficient(torch.sigmoid(torch.tensor(img_out)), torch.tensor(img_gt)))


                        save_img(img_out, os.path.join(noise_dir, f'output_{start_idx + j}.png'))
                        save_img(img_gt, os.path.join(noise_dir, f'output_{start_idx + j}y.png'))



            # Log outputs for uniqueness check (compare 5 random outputs)
            if len(unique_outputs) >= 5:
                print(f"Sample output pixel values for cell {cell_id} at noise level {noise_level}:")
                for sample_idx in range(5):
                    print(f"Sample {sample_idx + 1}: {unique_outputs[sample_idx].flatten()[:10]}")

            # Plot and save Dice scores
            plt.plot(dice_scores)
            plt.xlabel('Time Frame')
            plt.ylabel('Dice Score')
            plt.savefig(os.path.join(noise_dir, 'dice_score.png'))
            
            # Save Dice scores to CSV and print mean
            np.savetxt(os.path.join(noise_dir, 'dice_scores.csv'), dice_scores, delimiter=',')
            mean_dice_score = np.mean(dice_scores)
            print(f"Mean Dice score for cell {cell_id} with noise level {noise_level}: {mean_dice_score}")
            dice_score_by_cell[cell_id] = mean_dice_score

    # Summary of Dice scores across all cells
    print("\nSummary of mean Dice scores across cells:")
    for cell_id, mean_dice in dice_score_by_cell.items():
        print(f"Cell ID {cell_id}: Mean Dice Score = {mean_dice}")

    print("Output images and metrics saved successfully.")

#save the x_test_dialated_per_kernel and y_test_dialated_per_kernel
with open(DATA_PATH + 'x_test_dialated_per_kernel.pkl', 'wb') as f:
    pickle.dump(x_test_dialated_per_kernel, f)

with open(DATA_PATH + 'y_test_dialated_per_kernel.pkl', 'wb') as f:
    pickle.dump(y_test_dialated_per_kernel, f)
   
