import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

# --- 1. The Encoder Architecture ---
# This is a 1D Convolutional Neural Network that takes a 1-second EEG segment
# and compresses it into a feature vector (representation).

class EEGEncoder(nn.Module):
    def __init__(self, num_channels=16, representation_dim=128):
        """
        Args:
            num_channels (int): Number of EEG channels in the input.
            representation_dim (int): The dimension of the output feature vector 'h'.
        """
        super(EEGEncoder, self).__init__()
        self.num_channels = num_channels
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # At this point, the sequence length is 512 / (2*2*2*2) = 32
        )
        
        # A simple linear layer to get the final representation
        # The input size depends on the output of the convolutional layers
        # For a 512Hz input, final sequence length is 32. So, 256 * 32
        self.fc = nn.Linear(256 * 32, representation_dim)

    def forward(self, x):
        """
        Input shape: (batch_size, num_channels, sequence_length)
        e.g., (B, 16, 512)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        h = self.fc(x) # 'h' is the representation vector
        return h


# --- 2. The Projection Head ---
# As shown in the SimCLR paper, adding a non-linear projection head on top of the
# encoder during training improves the quality of the learned representations.
# We will ONLY use the encoder 'h' for downstream tasks, not the projection 'z'.

class ProjectionHead(nn.Module):
    def __init__(self, representation_dim=128, projection_dim=128):
        """
        Args:
            representation_dim (int): The dimension of the input feature vector 'h'.
            projection_dim (int): The dimension of the output embedding 'z'.
        """
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(representation_dim, projection_dim)
        )

    def forward(self, h):
        """
        Input shape: (batch_size, representation_dim)
        """
        z = self.head(h) # 'z' is the embedding used for contrastive loss
        return z


# --- 3. The Full Contrastive Model ---

class ContrastiveModel(nn.Module):
    def __init__(self, encoder, head):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        h = self.encoder(x)
        z = self.head(h)
        return z


# --- 4. Augmentation Module ---
# This is a critical component. We define a set of transformations to create
# two correlated "views" of the same EEG sample.

def eeg_augment(batch_tensor, noise_level=0.1, scale_range=(0.8, 1.2), mask_p=0.2):
    """
    Applies a random augmentation from a set of possible ones.
    Args:
        batch_tensor (torch.Tensor): The input EEG data (B, C, L).
    """
    aug_type = random.choice(['noise', 'scale', 'mask', 'none'])
    
    if aug_type == 'noise':
        noise = torch.randn_like(batch_tensor) * noise_level
        return batch_tensor + noise
    
    elif aug_type == 'scale':
        scaler = torch.rand(batch_tensor.size(0), 1, 1, device=batch_tensor.device) * \
                 (scale_range[1] - scale_range[0]) + scale_range[0]
        return batch_tensor * scaler

    elif aug_type == 'mask':
        # Mask a portion of the time series
        mask = torch.ones_like(batch_tensor)
        for i in range(batch_tensor.size(0)):
            mask_len = int(batch_tensor.size(2) * mask_p)
            mask_start = random.randint(0, batch_tensor.size(2) - mask_len)
            mask[i, :, mask_start:mask_start+mask_len] = 0
        return batch_tensor * mask
    
    else: # 'none'
        return batch_tensor


# --- 5. NT-Xent Loss Function ---
# The normalized temperature-scaled cross-entropy loss, which is the
# standard loss function for SimCLR-style contrastive learning.

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, device='cpu'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """
        z_i and z_j are the two augmented views of the same batch.
        Shape: (batch_size, projection_dim)
        """
        batch_size = z_i.shape[0]
        
        # Concatenate all embeddings
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Calculate the cosine similarity matrix
        sim_matrix = self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0))
        
        # Create labels for positive pairs
        # Positive pairs are (i, i+N) and (i+N, i)
        l_pos = torch.diag(sim_matrix, batch_size)
        r_pos = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        
        # Mask out self-similarity from the negative pairs
        diag = torch.eye(2 * batch_size, device=self.device, dtype=torch.bool)
        diag[diag.clone()] = 0
        
        negatives = sim_matrix[~diag].view(2 * batch_size, -1)
        
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        
        # Labels are always the first column (the positive examples)
        labels = torch.zeros(2 * batch_size).to(self.device).long()
        
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)


# --- 6. Dataset and Training Loop ---

class EEGDataset(Dataset):
    def __init__(self, data_array):
        """
        Args:
            data_array (np.array): A numpy array of shape (num_samples, num_channels, sequence_length)
                                   e.g., (1000000, 16, 512)
        """
        self.data = torch.from_numpy(data_array).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    # --- Configuration ---
    NUM_CHANNELS = 16
    SEQ_LENGTH = 512
    REPRESENTATION_DIM = 128
    PROJECTION_DIM = 64
    
    BATCH_SIZE = 512
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    TEMPERATURE = 0.07
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- Create Dummy Data ---
    # In your case, you would load your actual EEG data here.
    # The data should be pre-processed into 1-second segments.
    print("Creating dummy data...")
    num_dummy_samples = BATCH_SIZE * 10 
    dummy_eeg_data = np.random.randn(num_dummy_samples, NUM_CHANNELS, SEQ_LENGTH)
    
    # --- Setup Dataset and DataLoader ---
    dataset = EEGDataset(dummy_eeg_data)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --- Initialize Model, Loss, and Optimizer ---
    print("Initializing model...")
    encoder = EEGEncoder(num_channels=NUM_CHANNELS, representation_dim=REPRESENTATION_DIM)
    head = ProjectionHead(representation_dim=REPRESENTATION_DIM, projection_dim=PROJECTION_DIM)
    model = ContrastiveModel(encoder, head).to(DEVICE)
    
    loss_fn = NTXentLoss(temperature=TEMPERATURE, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, batch in enumerate(data_loader):
            batch = batch.to(DEVICE)
            
            # Create two augmented views of the batch
            batch_i = eeg_augment(batch)
            batch_j = eeg_augment(batch)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Get the embeddings
            z_i = model(batch_i)
            z_j = model(batch_j)
            
            # Calculate the loss
            loss = loss_fn(z_i, z_j)
            
            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # --- Save the Encoder ---
    # After training, you only need the encoder part for your downstream task.
    print("Training finished. Saving encoder weights...")
    torch.save(model.encoder.state_dict(), 'eeg_encoder.pth')
    print("Encoder saved to eeg_encoder.pth")
