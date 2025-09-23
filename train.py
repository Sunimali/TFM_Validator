# # Dice Loss & Metric
# def dice_loss(preds, targets, smooth=1e-6):
#     preds = torch.sigmoid(preds)
#     intersection = (preds * targets).sum()
#     union = preds.sum() + targets.sum() - intersection
#     return 1 - (2 * intersection + smooth) / (union + smooth)

# def dice_score(preds, targets, smooth=1e-6):
#     preds = (preds > 0.5).float()
#     return (2 * (preds * targets).sum() + smooth) / (preds.sum() + targets.sum() + smooth)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from networks import UNET  # Ensure this matches your model architecture
from utils import save_img  # Ensure this utility exists
import torch.nn.functional as F
import time

# Configuration
DATA_PATH = 'dataset/new/'
INITAL_LR = 0.0001
WEIGHT_DECAY = 1e-4
SAVE_EPOCH = 10
LOG_PATH = 'logs/dice_log_de_for_result.txt'
BATCH_SIZE = 10
NUM_EPOCHS = 150

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset Class
# class TFMDataset(Dataset):
#     def __init__(self, x_data, y_data, mean=None, std=None):
#         self.images = torch.from_numpy(np.array([x[0] for x in x_data])).float()
#         self.tractions = torch.from_numpy(np.array([x[1] for x in x_data])).float()
#         self.masks = torch.from_numpy(np.array(y_data) / 255.0).float()  
        
#         # Normalize
#         if mean is None or std is None:
#             self.mean, self.std = self.images.mean(), self.images.std()
#         else:
#             self.mean, self.std = mean, std

#         self.images = (self.images - self.mean) / self.std
#         self.tractions = (self.tractions - self.tractions.mean()) / self.tractions.std()
#         # self.images = (self.images - self.images.mean()) / self.images.std()
#         # self.tractions = (self.tractions - self.tractions.mean()) / self.tractions.std()

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         return (self.images[idx], self.tractions[idx]), self.masks[idx]
class TFMDataset(Dataset):
    def __init__(self, x_data, y_data, img_mean=None, img_std=None, trac_mean=None, trac_std=None):
        self.images = torch.from_numpy(np.array([x[0] for x in x_data])).float()
        self.tractions = torch.from_numpy(np.array([x[1] for x in x_data])).float()
        self.masks = torch.from_numpy(np.array(y_data) / 255.0).float()

        # Use training stats if provided
        self.img_mean = img_mean if img_mean is not None else self.images.mean()
        self.img_std = img_std if img_std is not None else self.images.std()
        self.trac_mean = trac_mean if trac_mean is not None else self.tractions.mean()
        self.trac_std = trac_std if trac_std is not None else self.tractions.std()

        # Normalize
        self.images = (self.images - self.img_mean) / self.img_std
        self.tractions = (self.tractions - self.trac_mean) / self.trac_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (self.images[idx], self.tractions[idx]), self.masks[idx]

# Load and Process Data
def load_data():
    with open(DATA_PATH + 'x_new_combined.pkl', 'rb') as f:
        x_data = pickle.load(f)
    with open(DATA_PATH + 'y_new_combined.pkl', 'rb') as f:
        y_data = pickle.load(f)

    # Create datasets
    train_dataset = TFMDataset(x_data['train'], y_data['train'])
    # test_dataset = TFMDataset(x_data['test'], y_data['test'], mean=train_dataset.mean, std=train_dataset.std)
    # Use training stats for both images and tractions
    test_dataset = TFMDataset(
        x_data['test'], y_data['test'],
        img_mean=train_dataset.img_mean, 
        img_std=train_dataset.img_std,
        trac_mean=train_dataset.trac_mean, 
        trac_std=train_dataset.trac_std
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)
    
    return train_loader, test_loader


# def dice_loss(preds, targets, smooth=1e-6):
#     # Remove thresholding and ensure targets are [0,1]
#     preds = torch.sigmoid(preds)  # Convert logits to probabilities
#     intersection = (preds * targets).sum()
#     denominator = preds.sum() + targets.sum()
#     return 1 - (2 * intersection + smooth) / (denominator + smooth)

# def dice_score(preds, targets, smooth=1e-6):
#     # Threshold only for evaluation
#     preds = (preds > 0.5).float()
#     targets = (targets > 0.5).float()  # Ensure binary targets
#     intersection = (preds * targets).sum()
#     denominator = preds.sum() + targets.sum()
#     return (2 * intersection + smooth) / (denominator + smooth)

def dice_loss(preds_logits, targets, smooth=1e-6):
    # Convert logits to probabilities
    preds = torch.sigmoid(preds_logits)
    
    # Flatten tensors
    preds_flat = preds.contiguous().view(-1)
    targets_flat = targets.contiguous().view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    denominator = preds_flat.sum() + targets_flat.sum()
    
    dice = (2. * intersection + smooth) / (denominator + smooth)
    return 1 - dice

def dice_score(preds_logits, targets, smooth=1e-6):
    """
    Dice Score for evaluation (thresholds predictions)
    Args:
        preds_logits: (N, H, W) - model raw outputs
        targets: (N, H, W) - ground truth [0,1] masks
    """
    # Convert to probabilities and threshold
    preds = torch.sigmoid(preds_logits)
    preds_bin = (preds > 0.5).float()
    targets_bin = (targets > 0.5).float()  # Ensure targets are binary
    
    # Flatten tensors
    preds_flat = preds_bin.contiguous().view(-1)
    targets_flat = targets_bin.contiguous().view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    denominator = preds_flat.sum() + targets_flat.sum()
    
    return (2. * intersection + smooth) / (denominator + smooth)

def composite_loss(preds_logits, targets, alpha=0.7, smooth=1e-6):
    """
    Combined BCE + Dice Loss
    Args:
        alpha: Weight for BCE (0.5-0.8 works best for most medical data)
    """
    # Binary Cross Entropy (handles logits)
    bce = F.binary_cross_entropy_with_logits(
        input=preds_logits, 
        target=targets,
        reduction='mean'
    )
    
    # Dice Loss
    dice = dice_loss(preds_logits, targets, smooth)
    
    return alpha * bce + (1 - alpha) * dice

# def composite_loss(preds_logits, targets, alpha=0.5):
#     bce = F.binary_cross_entropy_with_logits(preds_logits, targets)
#     preds = torch.sigmoid(preds_logits)
#     dice = dice_loss(preds, targets)
#     return alpha*bce + (1-alpha)*dice

# Training Setup
def initialize_model():
    model = UNET()
    model = nn.DataParallel(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=INITAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    return model, optimizer, scheduler

# Training Loop
def train_model():
    print("Loading data...")
    train_loader, test_loader = load_data()
    print("Data loaded successfully.")
    model, optimizer, scheduler = initialize_model()

    #load model
    # model.load_state_dict(torch.load('weights/new/best_model_vnv.pth'))
    
    best_dice = 0
    with open(LOG_PATH, 'w') as log_file:
        start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            # Training Phase
            model.train()
            train_loss = 0
            for batch_idx, ((images, tractions), masks) in enumerate(train_loader):
                images, tractions, masks = images.to(device), tractions.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model((images, tractions))
                loss = composite_loss(outputs, masks)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                # Time tracking
                elapsed_time = time.time() - start_time
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))

                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Elapsed: {elapsed_str}", end='\r')
                # print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end='\r')

            # Validation Phase
            model.eval()
            val_dice = 0

            
            with torch.no_grad():
                for (images, tractions), masks in test_loader:
                    images, tractions, masks = images.to(device), tractions.to(device), masks.to(device)
            
                    outputs = model((images, tractions))
                    val_dice += dice_score(outputs, masks).item() 

                # Save sample prediction

                save_img(outputs[0].cpu().numpy(), f'img/epoch_{epoch+1}_result.png')
                save_img(masks[0].cpu().numpy(), f'img/target.png')
            # Epoch Statistics
            train_loss /= len(train_loader)
            val_dice /= len(test_loader)
            scheduler.step(val_dice)

            # Logging to file
            log_file.write(f"\nEpoch {epoch+1}/{NUM_EPOCHS}\n")
            log_file.write(f"Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}\n")
            log_file.write(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}\n")


            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Save model
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), 'weights/new/de_time_best_model.pth')
                print(f"New best model saved with Dice: {best_dice:.4f}")

            if (epoch+1) % SAVE_EPOCH == 0:
                torch.save(model.state_dict(), f'weights/new/epoch_time_{epoch+1}.pth')

    torch.save(model.state_dict(), 'weights/new/de_time_final_model.pth')

if __name__ == '__main__':
    train_model()
    
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from networks import UNET  # Make sure this is properly imported
# from utils import save_img  # Make sure this is properly imported

# DATA_PATH = 'dataset/new/'
# # Hyperparameters
# INITAL_LR = 0.01
# LR_SCALER = 0.5
# STEP_LR = 15
# WEIGHT_DECAY = 1e-4
# SAVE_EPOCH = 10
# LOG_PATH = 'logs/dice_log.txt'

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch_device = torch.device(device)

# # Load data
# with open(DATA_PATH + 'x_new.pkl', 'rb') as f:
#     x_data = pickle.load(f)
    
# with open(DATA_PATH + 'y_new.pkl', 'rb') as f:
#     y_data = pickle.load(f)

# # Process data with proper train/test separation
# # ------------------------------------------------------------------
# x_img_train, x_trac_train = [], []
# for x_i in x_data['train']:
#     x_img_train.append(x_i[0])
#     x_trac_train.append(x_i[1])

# x_img_test, x_trac_test = [], []  # Corrected this section
# for x_i in x_data['test']:
#     x_img_test.append(x_i[0])  # Was previously adding to train
#     x_trac_test.append(x_i[1])  # Was previously adding to train

# # Convert to numpy arrays
# x_img_train = np.array(x_img_train)
# x_trac_train = np.array(x_trac_train)
# x_img_test = np.array(x_img_test)
# x_trac_test = np.array(x_trac_test)
# y_train = np.array(y_data['train'])
# y_test = np.array(y_data['test'])

# # Verify shapes
# print(f"Train image shape: {x_img_train.shape}")
# print(f"Train traction shape: {x_trac_train.shape}")
# print(f"Test image shape: {x_img_test.shape}")
# print(f"Test traction shape: {x_trac_test.shape}")
# # ------------------------------------------------------------------

# # Model setup
# model = UNET()
# model = nn.DataParallel(model).to(torch_device)

# # Convert to tensors
# x_img_train = torch.from_numpy(x_img_train).float()
# x_img_test = torch.from_numpy(x_img_test).float()
# x_trac_train = torch.from_numpy(x_trac_train).float()
# x_trac_test = torch.from_numpy(x_trac_test).float()
# y_train = torch.from_numpy(y_train).float()
# y_test = torch.from_numpy(y_test).float()

# # Normalize
# def normalize(tensor):
#     return (tensor - tensor.mean()) / tensor.std()

# x_img_train = normalize(x_img_train)
# x_img_test = normalize(x_img_test)
# x_trac_train = normalize(x_trac_train)
# x_trac_test = normalize(x_trac_test)

# def dice_score(preds, targets, smooth=1e-6):
#     preds = (preds > 0.5).float()
#     return (2 * (preds * targets).sum() + smooth) / (preds.sum() + targets.sum() + smooth)

# # Loss and optimizer
# def dice_loss(outputs, targets, smooth=1e-6):
#     probs = torch.sigmoid(outputs)
#     intersection = (probs * targets).sum()
#     union = probs.sum() + targets.sum()
#     return 1 - (2. * intersection + smooth) / (union + smooth)

# optimizer = optim.Adam(model.parameters(), 
#                       lr=INITAL_LR,
#                       weight_decay=WEIGHT_DECAY)
# scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)

# # Training parameters
# num_epochs = 150
# batch_size = 10
# clip_value = 5
# num_batches = len(x_img_train) // batch_size
# test_batches = len(x_img_test) // batch_size
# best_val_dice = -float('inf')
# best_model_path = 'weights/new/best_model.pth'

# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0

#     for i in range(num_batches):
#         start = i * batch_size
#         end = start + batch_size
        
#         img_batch = x_img_train[start:end].to(device)
#         trac_batch = x_trac_train[start:end].to(device)
#         target_batch = y_train[start:end].to(device)

#         optimizer.zero_grad()
#         outputs = model((img_batch, trac_batch))
#         loss = dice_loss(outputs, target_batch)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#         optimizer.step()
        
#         epoch_loss += loss.item()
#         print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}", end='\r')

#     # Validation
#     model.eval()
#     val_dice = 0
#     with torch.no_grad():
#         for i in range(test_batches):
#             start = i * batch_size
#             end = start + batch_size
            
#             img_val = x_img_test[start:end].to(device)
#             trac_val = x_trac_test[start:end].to(device)
#             target_val = y_test[start:end].to(device)
            
#             outputs = model((img_val, trac_val))
#             probs = torch.sigmoid(outputs)
#             val_dice += dice_score(probs, target_val).item()
            
#             if i == 0:
#                 save_img(probs[0].cpu().numpy(), f'img/new/epoch_{epoch+1}_result.png')

#     # Update scheduler
#     val_dice /= test_batches
#     scheduler.step(val_dice)
    
#     # Logging
#     train_loss = epoch_loss / num_batches
#     print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
    
#     with open(LOG_PATH, 'a') as f:
#         f.write(f"{train_loss:.4f},{val_dice:.4f}\n")
    
#     if val_dice > best_val_dice:
#         best_val_dice = val_dice
#         torch.save(model.state_dict(), best_model_path)
#         print(f"New best model saved with Dice: {val_dice:.4f}")

#     if (epoch+1) % SAVE_EPOCH == 0:
#         torch.save(model.state_dict(), f'weights/new/epoch_{epoch+1}.pth')

# torch.save(model.state_dict(), 'weights/new/final_model.pth')
# import pickle
# import numpy as np
# from sklearn.model_selection import train_test_split

# import torch
# import torch.optim as optim
# import torch.nn as nn

# from networks import UNET
# from utils import save_img
# # from custom_loss import SSIMLoss

# DATA_PATH = 'dataset/'
# INITAL_LR = 0.05
# LR_SCALER = 0.9
# STEP_LR = 25
# SAVE_EPOCH = 20
# LOG_PATH = 'logs/rmse.txt'
# WEIGHT_DECAY = 1e-5

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch_device = torch.device(device)

# with open(DATA_PATH + 'x_new.pkl', 'rb') as f:
#     x = pickle.load(f)
    
# with open(DATA_PATH + 'y_new.pkl', 'rb') as f:
#     y = np.array(pickle.load(f))

# x_img, x_trac = [], []
# for x_i in x:
#     x_img.append(x_i[0])
#     x_trac.append(x_i[1])

# x_img, x_trac = np.array(x_img), np.array(x_trac)

# x_img_train, x_img_test, x_trac_train, x_trac_test, y_train, y_test \
#     = train_test_split(x_img, x_trac, y, test_size=0.2)

# with open(DATA_PATH + 'x_img_train.pkl', 'wb') as f:
#     pickle.dump(x_img_train, f)
    
# with open(DATA_PATH + 'x_trac_train.pkl', 'wb') as f:
#     pickle.dump(x_trac_train, f)

# with open(DATA_PATH + 'y_train.pkl', 'wb') as f:
#     pickle.dump(y_train, f)

# with open(DATA_PATH + 'x_img_test.pkl', 'wb') as f:
#     pickle.dump(x_img_test, f)

# with open(DATA_PATH + 'x_trac_test.pkl', 'wb') as f:
#     pickle.dump(x_trac_test, f)
# with open(DATA_PATH + 'y_test.pkl', 'wb') as f:
#     pickle.dump(y_test, f)

   
# model = UNET()

# # comment to disable GPU training
# model = nn.DataParallel(model).to(torch_device)

# # Convert the data to PyTorch tensors
# x_img_train = torch.from_numpy(x_img_train).float()
# x_img_test = torch.from_numpy(x_img_test).float()
# x_trac_train = torch.from_numpy(x_trac_train).float()
# x_trac_test = torch.from_numpy(x_trac_test).float()
# y_train = torch.from_numpy(y_train).float()
# y_test = torch.from_numpy(y_test).float()

# # normalize the data
# x_img_train = (x_img_train - x_img_train.mean()) / x_img_train.std()
# x_img_test = (x_img_test - x_img_test.mean()) / x_img_test.std()
# x_trac_train = (x_trac_train - x_trac_train.mean()) / x_trac_train.std()
# x_trac_test = (x_trac_test - x_trac_test.mean()) / x_trac_test.std()

# y_train = (y_train - y_train.mean()) / y_train.std()
# y_test = (y_test - y_test.mean()) / y_test.std()

# # Define the loss function and optimizer
# criterion = torch.nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=INITAL_LR, weight_decay=WEIGHT_DECAY)

# # Train the model
# num_epochs = 150
# batch_size = 10
# clip_value = 5
# num_batches = len(x_img_train) // batch_size
# test_batches = len(x_img_test) // batch_size
# best_val_rmse = float('inf')
# best_model_path = 'weights/best_model.pth'

# # save target image
# # save_img(y_test[0].detach().cpu().numpy(), 'img/target.png')

# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0

#     for i in range(num_batches):
#         start_idx = i * batch_size
#         end_idx = (i + 1) * batch_size
        
#         # Forward pass
#         data_img = x_img_train[start_idx:end_idx].to(torch_device)
#         data_trac = x_trac_train[start_idx:end_idx].to(torch_device)
#         targets = y_train[start_idx:end_idx].to(torch_device)

#         outputs = model((data_img, data_trac))
#         optimizer.zero_grad()
#         loss = criterion(outputs, targets)
#         epoch_loss += loss.item()
        
#         # Backward pass and optimization
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#         optimizer.step()
        
#         print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}", end='\r')

#     # Learning rate scheduler
#     if (epoch+1) % STEP_LR == 0:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] *= LR_SCALER
    
#     # Evaluate the model on the test set
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for i in range(test_batches):
#             start_idx = i * batch_size
#             end_idx = (i + 1) * batch_size

#             test_data_img = x_img_test[start_idx:end_idx].to(torch_device)
#             test_data_trac = x_trac_test[start_idx:end_idx].to(torch_device)
#             tgt_test = y_test[start_idx:end_idx].to(torch_device)

#             outputs = model((test_data_img, test_data_trac))
#             val_loss += criterion(outputs, tgt_test)
            
#             if i == 0:
#                 # Save the output as an image
#                 output_image = outputs[0].detach().cpu().numpy()
#                 save_img(output_image, f'img/epoch_{epoch + 1}_result.png')
    
#     train_rmse = np.sqrt(epoch_loss/num_batches)
#     val_rmse = np.sqrt(val_loss.cpu()/test_batches)
#     print(f"Epoch {epoch+1}/{num_epochs}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}                ")
    
#     with open(LOG_PATH, 'a') as f:
#         f.write(f"{train_rmse:.4f}, {val_rmse:.4f}\n")
    
#     # Save model weights if validation RMSE improves
#     if val_rmse < best_val_rmse:
#         best_val_rmse = val_rmse
#         torch.save(model.state_dict(), best_model_path)
#         print(f"New best model saved with Val RMSE: {best_val_rmse:.4f}")

    
#     if epoch == SAVE_EPOCH:
#         torch.save(model.state_dict(), f'weights/mid_inj_{epoch+1}.pth')

# # Save model weights
# torch.save(model.state_dict(), 'weights/mid_inj.pth')
