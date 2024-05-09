# import requests
# from PIL import Image
# import open_clip
# import torch
# from transformers import AutoTokenizer
# from legrad import LeWrapper, LePreprocess, visualize
# from token_alignment import align_tokens  # Import the alignment function
#
# # Setup
# model_name = 'ViT-B-16'
# pretrained = 'laion2b_s34b_b88k'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
# open_tokenizer = open_clip.get_tokenizer(model_name=model_name)
# tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
# model = LeWrapper(model).to(device)
# preprocess = LePreprocess(preprocess=preprocess, image_size=448)
#
# # Process image
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)
#
# # Align and use tokens
# text = ['a photo of a cat', 'a photo of a remote control']
# aligned_tokens = align_tokens(open_tokenizer, tokenizer, text)
# # print(aligned_tokens)
# # print(open_tokenizer)
# # print(tokenizer)
# print(aligned_tokens)
# first_tokens = aligned_tokens[1]
# input_ids = torch.tensor(first_tokens, dtype=torch.long).unsqueeze(0).to(device)  # Adjust dimensions as necessary
#
# try:
#     text_embedding = model.encode_text(input_ids, normalize=True)
#     explainability_map = model.compute_legrad_clip(image=image, text_embedding=text_embedding)
#     visualize(heatmaps=explainability_map, image=image)
# except RuntimeError as e:
#     print("Runtime error:", e)



import requests
from PIL import Image
import open_clip
import torch
from transformers import AutoTokenizer
from legrad import LeWrapper, LePreprocess, visualize
from token_alignment import align_tokens  # Import the alignment function
import torch.nn.functional as F

# Setup
model_name = 'ViT-B-16'
pretrained = 'laion2b_s34b_b88k'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
open_tokenizer = open_clip.get_tokenizer(model_name=model_name)
tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model_legrad = LeWrapper(model).to(device)
print(model_legrad)
preprocess = LePreprocess(preprocess=preprocess, image_size=448)

# Process image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

# Align and use tokens
text = ['a photo of a cat', 'a photo of a remote control']
aligned_tokens = align_tokens(open_tokenizer, tokenizer, text)
first_tokens = aligned_tokens[0]  # Assuming you want to process the first set of tokens

# Ensure the tensor has the correct length (77) expected by the model
input_ids = torch.tensor(first_tokens, dtype=torch.long).unsqueeze(0).to(device)
pad_size = 77 - input_ids.shape[1]  # Calculate how much padding is needed
if pad_size > 0:
    input_ids = F.pad(input_ids, (0, pad_size), "constant", 0)  # Pad at the end

try:
    text_embedding = model_legrad.encode_text(input_ids, normalize=True)
    explainability_map = model_legrad.compute_legrad_clip(image=image, text_embedding=text_embedding)
    visualize(heatmaps=explainability_map, image=image)
except RuntimeError as e:
    print("Runtime error:", e)

