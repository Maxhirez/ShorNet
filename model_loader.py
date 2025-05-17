import argparse
import os
import numpy as np
import torch
import collections
from torch.utils.data import DataLoader

# Import necessary classes from your main scripts
# Replace with your actual imports
from pytorch_prime_factorization import PrimeFactorMLP, PrimeFactorDataset, _tensor_to_int


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
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Infer architecture from checkpoint
    architecture = infer_model_architecture(checkpoint_path)
    print(f"Detected model architecture: {architecture}")
    
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
    
    return model, architecture


def test_model(model_path, data_file, num_samples=10, chunk_size=None, device=None):
    """Test the trained model on samples from the dataset."""
    # Determine device
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model and detect architecture
    model, architecture = load_model_from_checkpoint(model_path, device)
    model.eval()
    
    # If chunk size wasn't provided, try to estimate it
    if chunk_size is None:
        # Estimate chunk size based on input_dim and typical bit size
        input_bits = 1024  # Typical for this application
        chunk_size = input_bits // architecture['input_dim']
        print(f"Estimated chunk size: {chunk_size}")
    
    # Load dataset
    dataset = PrimeFactorDataset(
        data_file,
        input_bits=1024,
        output_bits=512,
        chunk_size=chunk_size
    )
    
    # Verify dimensions match
    if dataset.input_dim != architecture['input_dim']:
        print(f"Warning: Dataset input dimension ({dataset.input_dim}) doesn't match model input dimension ({architecture['input_dim']})")
        print("This might indicate a different chunk size was used during training.")
        print(f"Adjusting dataset to use chunk size that gives {architecture['input_dim']} input dimensions...")
        
        # Recalculate chunk size to match model
        adjusted_chunk_size = 1024 // architecture['input_dim']
        print(f"Adjusted chunk size: {adjusted_chunk_size}")
        
        # Reload dataset with adjusted chunk size
        dataset = PrimeFactorDataset(
            data_file,
            input_bits=1024,
            output_bits=512,
            chunk_size=adjusted_chunk_size
        )
    
    # Create a small test set
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    test_data = [dataset[i] for i in indices]
    
    print(f"Testing model on {num_samples} samples...")
    
    # Track accuracy metrics
    relative_errors_p = []
    relative_errors_q = []
    product_matches = 0
    
    chunk_size_to_use = dataset.chunk_size
    
    for i, (n_tensor, p_tensor, q_tensor) in enumerate(test_data):
        # Get true values
        true_n = _tensor_to_int(n_tensor, chunk_size_to_use)
        true_p = _tensor_to_int(p_tensor, chunk_size_to_use)
        true_q = _tensor_to_int(q_tensor, chunk_size_to_use)
        
        # Predict
        with torch.no_grad():
            n_tensor = n_tensor.to(device, dtype=torch.float32).unsqueeze(0)
            p_pred, q_pred = model(n_tensor)
            pred_p = _tensor_to_int(p_pred[0].cpu(), chunk_size_to_use)
            pred_q = _tensor_to_int(q_pred[0].cpu(), chunk_size_to_use)
        
        # Calculate product of predictions
        pred_product = pred_p * pred_q
        
        # Calculate errors
        rel_error_p = abs(pred_p - true_p) / true_p
        rel_error_q = abs(pred_q - true_q) / true_q
        
        relative_errors_p.append(rel_error_p)
        relative_errors_q.append(rel_error_q)
        
        # Check if product matches
        product_match = abs(pred_product - true_n) / true_n < 1e-6
        if product_match:
            product_matches += 1
        
        # Print results
        print(f"\nSample {i+1}:")
        print(f"True N (hex): 0x{true_n:0256x}")
        print(f"True P (hex): 0x{true_p:0128x}")
        print(f"True Q (hex): 0x{true_q:0128x}")
        print(f"Pred P (hex): 0x{pred_p:0128x}")
        print(f"Pred Q (hex): 0x{pred_q:0128x}")
        print(f"Pred N (hex): 0x{pred_product:0256x}")
        print(f"Relative Error P: {rel_error_p:.8f}")
        print(f"Relative Error Q: {rel_error_q:.8f}")
        print(f"Product Match: {product_match}")
        
        # Calculate bits that match
        p_match_bits = count_matching_bits(true_p, pred_p)
        q_match_bits = count_matching_bits(true_q, pred_q)
        print(f"Matching bits in P: {p_match_bits}/512 ({p_match_bits/5.12:.1f}%)")
        print(f"Matching bits in Q: {q_match_bits}/512 ({q_match_bits/5.12:.1f}%)")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Average Relative Error P: {np.mean(relative_errors_p):.8f}")
    print(f"Average Relative Error Q: {np.mean(relative_errors_q):.8f}")
    print(f"Product Match Rate: {product_matches/num_samples:.2%}")


def count_matching_bits(a, b):
    """Count how many bits match between two integers."""
    xor = a ^ b  # XOR will have 0 bits where bits match
    count = 0
    for i in range(512):  # Assuming 512-bit numbers
        if (xor & (1 << i)) == 0:
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Model Loader for Prime Factorization")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to test')
    parser.add_argument('--chunk-size', type=int, default=None, help='Bit chunk size used in training (will be auto-detected if not provided)')
    
    args = parser.parse_args()
    test_model(args.model, args.data, args.samples, args.chunk_size)
