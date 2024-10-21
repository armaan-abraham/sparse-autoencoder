from utils import model, AutoEncoder, cfg, data_dir, load_encoder_training_data, d_vocab
import torch
import tqdm
import gc
from pathlib import Path

data_dir = Path(__file__).parent / "analysis_data"

torch.set_grad_enabled(False)

model_name = "first_run"
encoder = AutoEncoder.load(model_name)

all_tokens = load_encoder_training_data()

device = cfg.device

batch_size = 1024    
total_tokens = int(1e7)
seq_len = cfg.seq_len
tokens_per_batch = batch_size * seq_len
num_batches = total_tokens // tokens_per_batch

# Number of features
num_features = cfg.dict_size

# Prepare to accumulate statistics
numerator = torch.zeros(num_features, d_vocab, device=device, dtype=torch.float64)
denominator = torch.zeros(num_features, device=device, dtype=torch.float64)
total_samples = 0

# Process tokens in batches
print("Processing tokens in batches...")
for batch_idx in tqdm.trange(num_batches):
    # Get a batch of tokens
    batch_tokens = all_tokens[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device)

    # Run the model and get activations and outputs
    with torch.no_grad():
        model_outputs, cache = model.run_with_cache(batch_tokens)
        model_outputs = torch.nn.functional.softmax(model_outputs.reshape(-1, d_vocab), dim=-1)

    # Get the feature activations
    acts = cache[cfg.act_name]
    acts = acts.reshape(-1, cfg.act_size)  # Shape: (batch_size * seq_len, act_size)
    _, _, feature_activations, _, _ = encoder(acts)
    feature_activations = feature_activations[:, :num_features]

    # Accumulate weighted sums
    numerator += feature_activations.T @ model_outputs  # Shape: (num_features, d_vocab)
    denominator += feature_activations.sum(dim=0)               # Shape: (num_features,)
    total_samples += feature_activations.size(0)

    # Clean up
    del batch_tokens, cache, acts, feature_activations, model_outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    # Update progress bar with GPU memory usage
    gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    tqdm.tqdm.set_postfix_str(f"GPU Memory: {gpu_memory_allocated:.2f} GB")


del model

print(f"Total samples processed: {total_samples}")

if torch.any(denominator == 0):
    print("Warning: Some features have no activations")

# Compute the average model output distributions per feature
feature_distributions = numerator / denominator.unsqueeze(1)  # Shape: (num_features, d_vocab)

# assert that each feature's model output distribution sums to 1
print(f"Number of features that do not sum to 1: {torch.sum(feature_distributions.sum(dim=1) < (1 - 1e-6))}")

# Ensure that each feature's model output distribution sums to 1
feature_distributions = feature_distributions / feature_distributions.sum(dim=1, keepdim=True)

# Save the feature distributions
feature_distributions_path = data_dir / f"{model_name}_feature_output_distributions.pt"
torch.save(feature_distributions.cpu(), feature_distributions_path)
print(f"Feature distributions saved to {feature_distributions_path}")

# Compute the pairwise Jensen-Shannon divergence between features using PyTorch
print("Computing pairwise Jensen-Shannon divergence between features...")
eps = 1e-8  # Small value to avoid log(0)
batch_size = 100  # Adjust based on your GPU memory
n_features = feature_distributions.size(0)
jsd_matrix = torch.zeros(n_features, n_features, device=device, dtype=torch.float32)

for i_start in tqdm.trange(0, n_features, batch_size):
    i_end = min(i_start + batch_size, n_features)
    p_batch = feature_distributions[i_start:i_end, :] + eps  # Shape: (batch_i_size, d_vocab)
    log_p = torch.log(p_batch)  # Shape: (batch_i_size, d_vocab)

    for j_start in range(0, n_features, batch_size):
        j_end = min(j_start + batch_size, n_features)
        q_batch = feature_distributions[j_start:j_end, :] + eps  # Shape: (batch_j_size, d_vocab)
        log_q = torch.log(q_batch)  # Shape: (batch_j_size, d_vocab)

        # Compute m = 0.5 * (p + q)
        m = 0.5 * (p_batch.unsqueeze(1) + q_batch.unsqueeze(0))  # Shape: (batch_i_size, batch_j_size, d_vocab)
        log_m = torch.log(m)

        # Compute KL(p || m)
        kl_p_m = torch.sum(p_batch.unsqueeze(1) * (log_p.unsqueeze(1) - log_m), dim=2)  # Shape: (batch_i_size, batch_j_size)
        # Compute KL(q || m)
        kl_q_m = torch.sum(q_batch.unsqueeze(0) * (log_q.unsqueeze(0) - log_m), dim=2)  # Shape: (batch_i_size, batch_j_size)

        # Compute JSD
        jsd_batch = 0.5 * (kl_p_m + kl_q_m)  # Shape: (batch_i_size, batch_j_size)

        # Fill in the JSD matrix
        jsd_matrix[i_start:i_end, j_start:j_end] = jsd_batch

    # Update progress bar with GPU memory usage
    gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    tqdm.tqdm.set_postfix_str(f"GPU Memory: {gpu_memory_allocated:.2f} GB")

# Move JSD matrix to CPU and convert to float32
jsd_matrix = jsd_matrix.cpu().float()

# Save the JSD matrix to file
jsd_matrix_path = data_dir / f"{model_name}_feature_output_jsd_matrix.pt"
torch.save(jsd_matrix, jsd_matrix_path)
print(f"JSD matrix saved to {jsd_matrix_path}")
