from utils import AutoEncoder
import torch
import einops
from pathlib import Path

data_dir = Path(__file__).parent / "analysis_data"

# Load the encoder and model configuration
model_name = "first_run"
encoder = AutoEncoder.load(model_name)

feature_directions = encoder.W_dec
cosine_similarities = einops.einsum(feature_directions, feature_directions, "d_dict_a d_model, d_dict_b d_model -> d_dict_a d_dict_b")

# save cosine similarities
torch.save(cosine_similarities, data_dir / f"{model_name}_cosine_similarities.pt")
