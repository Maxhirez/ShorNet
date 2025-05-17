import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import model and dataset classes from the main script
from pytorch_prime_factorization import PrimeFactorMLP, PrimeFactorDataset, _tensor_to_int


def test_model(model_path, data_file, num_samples=10, chunk_size=32, device=None):
    """Test the trained model on samples from the dataset."""
    # Determine device
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load dataset
    dataset = PrimeFactorDataset(
        data_file,
        input_bits=1024,
        output_bits=512,
        chunk_size=chunk_size
    )
    
    # Create a small test set
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    test_data = [dataset[i] for i in indices]
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = PrimeFactorMLP(
        input_dim=dataset.input_dim,
        hidden_dim=1024,  # This should match your training setup
        output_dim=dataset.output_dim,
        num_layers=6  # This should match your training setup
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Testing model on {num_samples} samples...")
    
    # Track accuracy metrics
    relative_errors_p = []
    relative_errors_q = []
    product_matches = 0
    
    for i, (n_tensor, p_tensor, q_tensor) in enumerate(test_data):
        # Get true values
        true_n = _tensor_to_int(n_tensor, chunk_size)
        true_p = _tensor_to_int(p_tensor, chunk_size)
        true_q = _tensor_to_int(q_tensor, chunk_size)
        
        # Predict
        with torch.no_grad():
            n_tensor = n_tensor.to(device, dtype=torch.float32).unsqueeze(0)
            p_pred, q_pred = model(n_tensor)
            pred_p = _tensor_to_int(p_pred[0].cpu(), chunk_size)
            pred_q = _tensor_to_int(q_pred[0].cpu(), chunk_size)
        
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
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Average Relative Error P: {np.mean(relative_errors_p):.8f}")
    print(f"Average Relative Error Q: {np.mean(relative_errors_q):.8f}")
    print(f"Product Match Rate: {product_matches/num_samples:.2%}")


def generate_synthetic_dataset(output_file, num_samples=1000, min_bits_p=480, max_bits_p=512, 
                              min_bits_q=480, max_bits_q=512):
    """Generate a synthetic dataset of large integers and their prime factors."""
    from sympy import randprime
    
    data = []
    
    print(f"Generating {num_samples} synthetic samples...")
    for _ in tqdm(range(num_samples)):
        # Generate random bit sizes for p and q
        p_bits = np.random.randint(min_bits_p, max_bits_p + 1)
        q_bits = np.random.randint(min_bits_q, max_bits_q + 1)
        
        # Generate random prime numbers with specified bit length
        p = randprime(2**(p_bits-1), 2**p_bits - 1)
        q = randprime(2**(q_bits-1), 2**q_bits - 1)
        
        # Calculate product
        n = p * q
        
        # Add to dataset
        data.append({'n': hex(n), 'p': hex(p), 'q': hex(q)})
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved {num_samples} samples to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prime Factorization Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate dataset command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic dataset')
    gen_parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    gen_parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    
    # Test model command
    test_parser = subparsers.add_parser('test', help='Test trained model')
    test_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    test_parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    test_parser.add_argument('--samples', type=int, default=10, help='Number of samples to test')
    test_parser.add_argument('--chunk-size', type=int, default=32, help='Bit chunk size used in training')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_synthetic_dataset(args.output, args.samples)
    elif args.command == 'test':
        test_model(args.model, args.data, args.samples, args.chunk_size)
    else:
        parser.print_help()
