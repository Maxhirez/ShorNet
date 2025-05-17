#!/usr/bin/env python3
"""
Simple inference script for the Prime Factorization MLP model.
Takes a 1024-bit hex number and predicts its prime factors.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import time


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


def int_to_tensor(num, bits, chunk_size):
    """Convert integer to normalized tensor representation using chunks."""
    # Create a tensor of the appropriate shape
    chunks = bits // chunk_size
    tensor = torch.zeros(chunks)
    
    # Fill tensor with normalized chunk values
    for i in range(chunks):
        # Extract chunk_size bits starting from the least significant bits
        mask = (1 << chunk_size) - 1
        chunk_val = (num >> (i * chunk_size)) & mask
        # Normalize to [0, 1]
        tensor[chunks - i - 1] = chunk_val / ((1 << chunk_size) - 1)
        
    return tensor


def tensor_to_int(tensor, chunk_size):
    """Convert tensor representation back to integer."""
    num = 0
    for i, chunk_val in enumerate(tensor):
        # Denormalize and convert to integer
        val = int(round(chunk_val.item() * ((1 << chunk_size) - 1)))
        # Shift to the appropriate position
        num |= val << ((len(tensor) - i - 1) * chunk_size)
    return num


# Define model classes needed for loading the checkpoint
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


def load_model_from_checkpoint(checkpoint_path, device=None):
    """Load a model from checkpoint, automatically detecting architecture."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
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
    
    return model, chunk_size, architecture


def predict_factors(model, number_hex, chunk_size, device=None):
    """Predict prime factors for a given hex number."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Convert hex to integer
    if number_hex.startswith('0x'):
        number = int(number_hex, 16)
    else:
        number = int('0x' + number_hex, 16)
    
    # Check if input is valid 1024-bit number
    if number.bit_length() > 1024:
        raise ValueError(f"Input number exceeds 1024 bits (actual: {number.bit_length()} bits)")
    
    # Convert to tensor
    n_tensor = int_to_tensor(number, 1024, chunk_size)
    
    # Predict
    model.eval()
    with torch.no_grad():
        n_tensor = n_tensor.to(device, dtype=torch.float32).unsqueeze(0)
        p_pred, q_pred = model(n_tensor)
        pred_p = tensor_to_int(p_pred[0].cpu(), chunk_size)
        pred_q = tensor_to_int(q_pred[0].cpu(), chunk_size)
    
    # Calculate product to verify
    pred_product = pred_p * pred_q
    product_error = abs(pred_product - number) / number
    
    return pred_p, pred_q, product_error


def main():
    parser = argparse.ArgumentParser(description="Predict prime factors of a 1024-bit hex number")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--number", type=str, required=True, help="1024-bit hex number to factorize (with or without 0x prefix)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Import torch.nn only after checking arguments
    import torch.nn as nn
    
    # Load model
    model, chunk_size, architecture = load_model_from_checkpoint(args.model, device)
    
    # Time the prediction
    start_time = time.time()
    
    # Make prediction
    p, q, error = predict_factors(model, args.number, chunk_size, device)
    
    end_time = time.time()
    
    # Print results
    print("\nResults:")
    print(f"Input number: {args.number}")
    print(f"Predicted p: 0x{p:0128x}")
    print(f"Predicted q: 0x{q:0128x}")
    print(f"Reconstructed n: 0x{p*q:0256x}")
    print(f"Relative error: {error:.10f}")
    
    # Check if prediction is close
    if error < 1e-6:
        print("✅ Prediction successful! The product of the predicted factors matches the input.")
    else:
        print("❌ Prediction needs improvement. The product doesn't match the input exactly.")
    
    print(f"\nInference time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
