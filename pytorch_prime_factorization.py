import argparse
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ResidualBlock(nn.Module):
    """Residual block for the MLP."""
    
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.activation(self.linear1(x))
        x = self.norm2(x)
        x = self.linear2(x)
        return x + identity


class PrimeFactorMLP(nn.Module):
    """MLP with residual connections and two output heads for prime factorization."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Layer normalization before output
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        
        # Two separate output heads for p and q
        self.p_head = nn.Linear(hidden_dim, output_dim)
        self.q_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Process through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        x = self.activation(x)
        
        # Generate predictions through separate heads
        p_pred = self.p_head(x)
        q_pred = self.q_head(x)
        
        return p_pred, q_pred


class PrimeFactorDataset(Dataset):
    """Dataset for loading and processing large integer data."""
    
    def __init__(self, csv_path, input_bits=1024, output_bits=512, chunk_size=32):
        """
        Args:
            csv_path: Path to CSV file with n,p,q format
            input_bits: Bit length of the input number n
            output_bits: Bit length of output primes p and q
            chunk_size: Size of bit chunks to use for representation
        """
        self.csv_path = csv_path
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.chunk_size = chunk_size
        
        # Calculate dimensions
        self.input_dim = input_bits // chunk_size
        self.output_dim = output_bits // chunk_size
        
        # Load data
        self.data = pd.read_csv(csv_path, header=None, names=['n', 'p', 'q'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Convert hexadecimal strings to integers if needed
        if isinstance(row['n'], str) and row['n'].startswith('0x'):
            n = int(row['n'], 16)
            p = int(row['p'], 16)
            q = int(row['q'], 16)
        else:
            n = int(row['n'])
            p = int(row['p'])
            q = int(row['q'])
        
        # Convert to bit representation and chunk
        n_tensor = self._int_to_tensor(n, self.input_bits)
        p_tensor = self._int_to_tensor(p, self.output_bits)
        q_tensor = self._int_to_tensor(q, self.output_bits)
        
        return n_tensor, p_tensor, q_tensor
    
    def _int_to_tensor(self, num, bits):
        """Convert integer to normalized tensor representation using chunks."""
        # Create a tensor of the appropriate shape
        chunks = bits // self.chunk_size
        tensor = torch.zeros(chunks)
        
        # Fill tensor with normalized chunk values
        for i in range(chunks):
            # Extract chunk_size bits starting from the least significant bits
            mask = (1 << self.chunk_size) - 1
            chunk_val = (num >> (i * self.chunk_size)) & mask
            # Normalize to [0, 1]
            tensor[chunks - i - 1] = chunk_val / ((1 << self.chunk_size) - 1)
            
        return tensor


def train_one_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for n_tensors, p_tensors, q_tensors in tqdm(dataloader, desc="Training"):
        # Move data to device
        n_tensors = n_tensors.to(device, dtype=torch.float32)
        p_tensors = p_tensors.to(device, dtype=torch.float32)
        q_tensors = q_tensors.to(device, dtype=torch.float32)
        
        # Forward pass
        p_pred, q_pred = model(n_tensors)
        
        # Calculate loss (MSE for each prime factor)
        p_loss = nn.functional.mse_loss(p_pred, p_tensors)
        q_loss = nn.functional.mse_loss(q_pred, q_tensors)
        loss = p_loss + q_loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for n_tensors, p_tensors, q_tensors in tqdm(dataloader, desc="Validation"):
            # Move data to device
            n_tensors = n_tensors.to(device, dtype=torch.float32)
            p_tensors = p_tensors.to(device, dtype=torch.float32)
            q_tensors = q_tensors.to(device, dtype=torch.float32)
            
            # Forward pass
            p_pred, q_pred = model(n_tensors)
            
            # Calculate loss
            p_loss = nn.functional.mse_loss(p_pred, p_tensors)
            q_loss = nn.functional.mse_loss(q_pred, q_tensors)
            loss = p_loss + q_loss
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def predict_prime_factors(model, n_tensor, device, dataset):
    """Predict prime factors for a given integer."""
    model.eval()
    with torch.no_grad():
        n_tensor = n_tensor.to(device, dtype=torch.float32).unsqueeze(0)
        p_pred, q_pred = model(n_tensor)
        
        # Convert predictions back to integers
        p_int = _tensor_to_int(p_pred[0].cpu(), dataset.chunk_size)
        q_int = _tensor_to_int(q_pred[0].cpu(), dataset.chunk_size)
        
        return p_int, q_int


def _tensor_to_int(tensor, chunk_size):
    """Convert tensor representation back to integer."""
    num = 0
    for i, chunk_val in enumerate(tensor):
        # Denormalize and convert to integer
        val = int(round(chunk_val.item() * ((1 << chunk_size) - 1)))
        # Shift to the appropriate position
        num |= val << ((len(tensor) - i - 1) * chunk_size)
    return num


def main(args):
    # Set up device
    if args.gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and data loaders
    dataset = PrimeFactorDataset(
        args.data_file,
        input_bits=1024,
        output_bits=512,
        chunk_size=args.chunk_size
    )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = PrimeFactorMLP(
        input_dim=dataset.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=dataset.output_dim,
        num_layers=args.num_layers
    )
    model.to(device)
    
    # Create optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(train_loader)
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        tic = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        toc = time.time()
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Time: {toc - tic:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(output_dir, 'final_model.pth'))
    
    print(f"Training completed. Models saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a prime factorization MLP with PyTorch and MPS")
    parser.add_argument("--data-file", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension size")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of residual layers")
    parser.add_argument("--chunk-size", type=int, default=32, help="Bit chunk size for representation")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use MPS (GPU) acceleration if available")
    
    args = parser.parse_args()
    main(args)
