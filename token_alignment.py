import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer
import open_clip

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=0)

def align_tokens(openclip_tokenizer, vicuna_tokenizer, texts):
    aligned_tokens_list = []
    for text in texts:
        # Tokenize text using both tokenizers
        openclip_tokens = openclip_tokenizer.encode(text)
        vicuna_tokens = vicuna_tokenizer.encode(text, return_tensors='pt',
                                   padding='max_length',  # Pad all sentences to the model's max length
                                   truncation=True,  # Truncate to model's max length
                                   max_length=77).squeeze(0)  # Max length of Vicuna's model

        # Create a token alignment matrix
        A = np.zeros((len(openclip_tokens), len(vicuna_tokens)))
        for i, openclip_token in enumerate(openclip_tokens):
            for j, vicuna_token in enumerate(vicuna_tokens):
                openclip_token_tensor = torch.tensor([openclip_token], dtype=torch.float32)
                vicuna_token_tensor = torch.tensor([vicuna_token.item()], dtype=torch.float32)
                similarity = cosine_similarity(openclip_token_tensor, vicuna_token_tensor)
                A[i, j] = similarity.item()

        # Compute the optimal token alignment
        row_ind, col_ind = linear_sum_assignment(-A)
        aligned_vicuna_tokens = vicuna_tokens[col_ind]

        aligned_tokens_list.append(aligned_vicuna_tokens)

    return aligned_tokens_list

if __name__ == "__main__":
    openclip_tokenizer = open_clip.get_tokenizer(model_name='ViT-B-16')
    vicuna_tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    texts = ['a photo of a cat', 'a photo of a remote control']
    print(align_tokens(openclip_tokenizer, vicuna_tokenizer, texts))
