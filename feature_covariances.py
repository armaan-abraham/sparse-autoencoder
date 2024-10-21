from utils import all_tokens, model, AutoEncoder, cfg
import torch
import numpy as np
import tqdm
import gc
from pathlib import Path

data_dir = Path(__file__).parent / "analysis_data"

torch.set_grad_enabled(False)

model_name = "first_run"
encoder = AutoEncoder.load(model_name)

device = cfg.device

batch_size = 1024    
total_tokens = int(1e7)
seq_len = cfg.seq_len
tokens_per_batch = batch_size * seq_len
num_batches = total_tokens // tokens_per_batch

num_features = cfg.dict_size

print(f"Number of features: {num_features}")

# Prepare to accumulate statistics
sum_X = torch.zeros(num_features, device=device, dtype=torch.float64)
sum_X2 = torch.zeros(num_features, device=device, dtype=torch.float64)
sum_outer = torch.zeros(num_features, num_features, device=device, dtype=torch.float64)
total_samples = 0


# Process tokens in batches
print("Processing tokens in batches...")
for batch_idx in tqdm.trange(num_batches):
    # Get a batch of tokens
    batch_tokens = all_tokens[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device)

    # Run the model and get activations
    with torch.no_grad():
        _, cache = model.run_with_cache(batch_tokens)

    # Get the feature activations
    model_acts = cache[cfg.act_name].reshape(-1, cfg.act_size)
    feature_activations = (model_acts - encoder.b_dec) @ encoder.W_enc + encoder.b_enc

    # Update accumulators
    total_samples += feature_activations.size(0)
    sum_X += feature_activations.sum(dim=0)
    sum_X2 += (feature_activations ** 2).sum(dim=0)
    sum_outer += feature_activations.T @ feature_activations

    gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    tqdm.tqdm.set_postfix_str(f"GPU Memory: {gpu_memory_allocated:.2f} GB")
    # Clean up
    del batch_tokens, cache, feature_activations, model_acts
    torch.cuda.empty_cache()
    gc.collect()

print(f"Total samples processed: {total_samples}")

# Compute mean and standard deviation
mean_X = sum_X / total_samples
var_X = (sum_X2 / total_samples) - mean_X ** 2
std_X = torch.sqrt(var_X)

# Handle any zero standard deviations
std_X[std_X == 0] = 1e-6

# Compute covariance matrix
covariance_matrix = (sum_outer / total_samples) - torch.ger(mean_X, mean_X)

# Compute correlation matrix
correlation_matrix = covariance_matrix / torch.ger(std_X, std_X)

# Move to CPU and convert to float32
correlation_matrix = correlation_matrix.cpu().float()

# Save correlation matrix to file
correlation_matrix_path = data_dir / f"{model_name}_feature_correlations.pt"
torch.save(correlation_matrix, correlation_matrix_path)
print(f"Correlation matrix saved to {correlation_matrix_path}")
