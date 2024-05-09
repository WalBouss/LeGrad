import requests
from PIL import Image
import open_clip
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



from legrad import LeWrapper, LePreprocess, visualize

# ------- model's paramters -------
model_name = 'ViT-B-16'
pretrained = 'laion2b_s34b_b88k'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------- init model -------
model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
open_tokenizer = open_clip.get_tokenizer(model_name=model_name)
tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')

model.eval()
# ------- Equip the model with LeGrad -------
model = LeWrapper(model)
# ___ (Optional): Wrapper for Higher-Res input image ___
preprocess = LePreprocess(preprocess=preprocess, image_size=448)

# ------- init inputs: image +  text -------
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)
opentext = open_tokenizer(['a photo of a cat', 'a photo of a remote control']).to(device)

encoded_inputs = tokenizer(['a photo of a cat', 'a photo of a remote control'],
                           return_tensors='pt',
                           padding='max_length',  # Pad all sentences to the model's max length
                           truncation=True,       # Truncate to model's max length
                           max_length=77)         # Adjust to your model's expected input size

input_ids = encoded_inputs['input_ids'].to(device)
attention_mask = encoded_inputs['attention_mask'].to(device)

print(encoded_inputs)
print()




# If the model expects a specific sequence length, confirm and adjust the max_length accordingly.
print("Input IDs shape:", input_ids.shape)  # This should reflect (batch_size, 77)


# Continue with the model processing
try:
    text_embedding = model.encode_text(input_ids, normalize=True)
    print(image.shape)
    explainability_map = model.compute_legrad_clip(image=image, text_embedding=text_embedding)
    visualize(heatmaps=explainability_map, image=image)
except RuntimeError as e:
    print("Runtime error:", e)