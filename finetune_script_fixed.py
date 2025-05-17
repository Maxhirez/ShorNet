#!/usr/bin/env python3
"""
Fine-tuning script for the Prime Factorization MLP model.
Loads a pre-trained model and continues training on a new/expanded dataset.
"""

import argparse
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import json


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
    
    def __init__(self, csv_path, input_bits=1024, output_bits=512, chunk_size=32, skip_header=False, df=None):
        """
        Args:
            csv_path: Path to CSV file with n,p,q format (can be None if df is provided)
            input_bits: Bit length of the input number n
            output_bits: Bit length of output primes p and q
            chunk_size: Size of bit chunks to use for representation
            skip_header: Whether to skip the first row of the CSV file
            df: Optional DataFrame to use instead of loading from csv_path
        """
        self.csv_path = csv_path
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.chunk_size = chunk_size
        
        # Calculate dimensions
        self.input_dim = input_bits // chunk_size
        self.output_dim = output_bits // chunk_size
        
        # Either use provided DataFrame or load from CSV
        if df is not None:
            self.data = df
            print(f"Using provided DataFrame with {len(self.data)} samples")
        elif csv_path is not None:
            # Load data from file
            if skip_header:
                self.data = pd.read_csv(csv_path, header=0, names=['n', 'p', 'q'])
            else:
                self.data = pd.read_csv(csv_path, header=None, names=['n', 'p', 'q'])
            print(f"Loaded {len(self.data)} samples from {csv_path}")
        else:
            self.data = pd.DataFrame(columns=['n', 'p', 'q'])
            print("Created empty dataset (will be populated later)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            # Handle list of indices (for DataLoader workers)
            return [self[i] for i in idx]
            
        # Handle single index
        if not (isinstance(idx, int) or np.issubdtype(type(idx), np.integer)):
            raise TypeError(f"Index must be integer, got {type(idx)}")
            
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


class ChunkedCSVLoader:
    """Memory-efficient loader for large CSV files."""
    
    def __init__(self, csv_path, batch_size, chunk_size=100000, shuffle=True, 
                 input_bits=1024, output_bits=512, tensor_chunk_size=32, 
                 skip_header=False, num_workers=0, pin_memory=False):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.tensor_chunk_size = tensor_chunk_size
        self.skip_header = skip_header
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Get file size and estimate rows for progress tracking
        file_size = os.path.getsize(csv_path)
        
        # Read a small sample to estimate average row size
        sample_size = min(1000, chunk_size)
        sample_df = pd.read_csv(csv_path, nrows=sample_size, 
                               header=0 if skip_header else None)
        
        # Estimate total rows
        avg_row_bytes = file_size / len(sample_df) if len(sample_df) > 0 else 100
        self.estimated_total_rows = int(file_size / avg_row_bytes)
        print(f"Estimated total rows in dataset: {self.estimated_total_rows}")
        
        # Initialize counters
        self.rows_processed = 0
        
        # Current chunk data loader
        self.current_loader = None
        self.current_loader_iter = None
        
    def __iter__(self):
        # Reset counters
        self.rows_processed = 0
        
        # Create reader for chunks
        self.reader = pd.read_csv(
            self.csv_path, 
            chunksize=self.chunk_size,
            header=0 if self.skip_header else None,
            names=['n', 'p', 'q']
        )
        
        # Start with the first chunk
        self._load_next_chunk()
        
        return self
    
    def _load_next_chunk(self):
        """Load the next chunk of data if available."""
        try:
            # Get next chunk
            chunk = next(self.reader)
            self.rows_processed += len(chunk)
            
            # Create dataset from this chunk
            dataset = PrimeFactorDataset(
                csv_path=None,
                df=chunk,
                input_bits=self.input_bits,
                output_bits=self.output_bits,
                chunk_size=self.tensor_chunk_size
            )
            
            # Create data loader for this chunk
            self.current_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            
            # Get iterator
            self.current_loader_iter = iter(self.current_loader)
            
            return True
        except StopIteration:
            # No more chunks
            self.current_loader = None
            self.current_loader_iter = None
            return False
    
    def __next__(self):
        # Try to get batch from current loader
        if self.current_loader_iter is not None:
            try:
                return next(self.current_loader_iter)
            except StopIteration:
                # Current chunk exhausted, try to load next chunk
                if self._load_next_chunk():
                    # If successful, return batch from new chunk
                    return next(self.current_loader_iter)
                else:
                    # No more chunks
                    raise StopIteration
        else:
            # No loader
            raise StopIteration
    
    def __len__(self):
        # Estimate based on total rows and batch size
        return self.estimated_total_rows // self.batch_size


def infer_model_architecture(checkpoint_path):
    """Analyze a saved checkpoint to determine model architecture parameters."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Extract architecture information from the state dict
    architecture_info = {}
    
    # Find input dimension from first layer weights
    input_proj_weight = state_dict.get('input_proj.weight')
    if input_proj_weight is not None:
        architecture_info['input_dim'] = input_proj_weight.shape[1]
        architecture_info['hidden_dim'] = input_proj_weight.shape[0]
    
    # Find output dimension from output head weights
    p_head_weight = state_dict.get('p_head.weight')
    if p_head_weight is not None:
        architecture_info['output_dim'] = p_head_weight.shape[0]
    
    # Count number of residual blocks
    residual_blocks = 0
    for key in state_dict.keys():
        if 'residual_blocks' in key and 'linear1.weight' in key:
            block_num = int(key.split('.')[1])
            residual_blocks = max(residual_blocks, block_num + 1)
    
    architecture_info['num_layers'] = residual_blocks
    
    return architecture_info


def load_model_from_checkpoint(checkpoint_path, device=None):
    """Load a model from checkpoint, automatically detecting architecture."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Infer architecture from checkpoint
    architecture = infer_model_architecture(checkpoint_path)
    print(f"Detected model architecture: {architecture}")
    
    # Calculate chunk size based on input dimensions (assuming 1024-bit input)
    chunk_size = 1024 // architecture['input_dim']
    print(f"Using chunk size: {chunk_size}")
    
    # Create model with inferred architecture
    model = PrimeFactorMLP(
        input_dim=architecture['input_dim'],
        hidden_dim=architecture['hidden_dim'],
        output_dim=architecture['output_dim'],
        num_layers=architecture['num_layers']
    )
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Return model, chunk size and epoch (for continuing training)
    epoch = checkpoint.get('epoch', 0)
    
    return model, chunk_size, architecture, epoch


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    batches = 0
    
    # Progress tracking
    if hasattr(dataloader, '__len__'):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs if total_epochs else '?'}")
    else:
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for n_tensors, p_tensors, q_tensors in progress_bar:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping for stability
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})
    
    return total_loss / max(1, batches)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    batches = 0
    
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
            batches += 1
    
    return total_loss / max(1, batches)


def create_validation_dataset(data_file, val_size, chunk_size, skip_header=False):
    """Create a small validation dataset from the beginning of a file."""
    print(f"Creating validation dataset with {val_size} samples...")
    
    # Load just the validation set size
    val_data = pd.read_csv(
        data_file, 
        nrows=val_size,
        header=0 if skip_header else None,
        names=['n', 'p', 'q']
    )
    
    # Create dataset
    return PrimeFactorDataset(
        csv_path=None,
        df=val_data,
        input_bits=1024,
        output_bits=512,
        chunk_size=chunk_size
    )


def main(args):
    # Set up device
    if args.gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration")
    elif args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args to output directory
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load model from checkpoint
    model, chunk_size, architecture, start_epoch = load_model_from_checkpoint(args.model, device)
    
    # Create validation dataset (smaller size, load fully into memory)
    if args.val_data:
        # Use separate validation file
        val_dataset = PrimeFactorDataset(
            csv_path=args.val_data,
            input_bits=1024,
            output_bits=512,
            chunk_size=chunk_size,
            skip_header=args.skip_header
        )
    else:
        # Create small validation set from training data
        val_dataset = create_validation_dataset(
            args.data_file, 
            val_size=args.val_samples, 
            chunk_size=chunk_size,
            skip_header=args.skip_header
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_workers if args.val_workers is not None else args.num_workers,
        pin_memory=True
    )
    
    # Create training data loader (memory-efficient version for large datasets)
    train_loader = ChunkedCSVLoader(
        args.data_file,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        shuffle=True,
        tensor_chunk_size=chunk_size,
        skip_header=args.skip_header,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Load optimizer state if available and requested
    if args.continue_optimizer and 'optimizer_state_dict' in torch.load(args.model, map_location='cpu'):
        try:
            optimizer.load_state_dict(torch.load(args.model, map_location='cpu')['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Could not load optimizer state: {e}")
    
    # Create learning rate scheduler
    if args.use_cosine_scheduler:
        # For cosine scheduler, need to specify total steps
        # Estimate based on dataset size and epochs
        total_steps = (train_loader.estimated_total_rows // args.batch_size) * args.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=args.min_lr
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=args.min_lr
        )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        tic = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, start_epoch + args.epochs)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        toc = time.time()
        
        print(f"Epoch {epoch+1}/{start_epoch + args.epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Time: {toc - tic:.2f}s")
        
        # Update learning rate
        if args.use_cosine_scheduler:
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'architecture': architecture,
                'chunk_size': chunk_size
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'architecture': architecture,
                'chunk_size': chunk_size
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': start_epoch + args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'architecture': architecture,
        'chunk_size': chunk_size
    }, os.path.join(output_dir, 'final_model.pth'))
    
    print(f"Training completed. Models saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a prime factorization MLP with PyTorch and MPS")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint to fine-tune")
    parser.add_argument("--data-file", type=str, required=True, help="Path to CSV data file for training")
    parser.add_argument("--val-data", type=str, default=None, help="Path to CSV data file for validation (optional)")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate (should be lower for fine-tuning)")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--val-workers", type=int, default=None, help="Number of validation data loader workers (defaults to num-workers)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--skip-header", action="store_true", help="Skip header row in CSV files")
    parser.add_argument("--continue-optimizer", action="store_true", help="Continue with saved optimizer state")
    parser.add_argument("--use-cosine-scheduler", action="store_true", help="Use cosine annealing scheduler")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Number of rows to load at once from CSV")
    parser.add_argument("--val-samples", type=int, default=10000, help="Number of validation samples to use (when using training data for validation)")
    
    args = parser.parse_args()
    main(args)
