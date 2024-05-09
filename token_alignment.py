import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import open_clip

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=0)

# Load the OpenCLIP tokenizer
openclip_tokenizer = open_clip.get_tokenizer(model_name='ViT-B-16')

# Load the Vicuna tokenizer
vicuna_tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')

# Tokenize the text using OpenCLIP
openclip_tokens = openclip_tokenizer.encode('a photo of a cat')

# Tokenize the text using Vicuna
vicuna_tokens = vicuna_tokenizer.encode('a photo of a cat', return_tensors='pt').squeeze(0)  # Squeezing to remove batch dimension

# Create a token alignment matrix
A = np.zeros((len(openclip_tokens), len(vicuna_tokens)))

for i, openclip_token in enumerate(openclip_tokens):
    for j, vicuna_token in enumerate(vicuna_tokens):
        # Convert tokens to tensors and specify the data type as float
        openclip_token_tensor = torch.tensor([openclip_token], dtype=torch.float32)  # Wrap scalar in a list to create a tensor
        vicuna_token_tensor = torch.tensor([vicuna_token.item()], dtype=torch.float32)  # Convert from tensor to scalar and back to tensor
        similarity = cosine_similarity(openclip_token_tensor, vicuna_token_tensor)
        A[i, j] = similarity.item()  # Convert tensor to scalar

# Compute the optimal token alignment
row_ind, col_ind = linear_sum_assignment(-A)

# Align the Vicuna tokens
aligned_vicuna_tokens = vicuna_tokens[col_ind]

print(aligned_vicuna_tokens)