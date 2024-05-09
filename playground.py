import requests
from PIL import Image
import open_clip
import torch

from legrad import LeWrapper, LePreprocess, visualize

# ------- model's paramters -------
model_name = 'ViT-B-16'
pretrained = 'laion2b_s34b_b88k'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------- init model -------
model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
tokenizer = open_clip.get_tokenizer(model_name=model_name)
model.eval()
# ------- Equip the model with LeGrad -------
model = LeWrapper(model)
# ___ (Optional): Wrapper for Higher-Res input image ___
preprocess = LePreprocess(preprocess=preprocess, image_size=448)

# ------- init inputs: image +  text -------
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)
text = tokenizer(['a photo of a cat', 'a photo of a remote control']).to(device)

# -------
text_embedding = model.encode_text(text, normalize=True)
print(image.shape)
explainability_map = model.compute_legrad_clip(image=image, text_embedding=text_embedding)

# ___ (Optional): Visualize overlay of the image + heatmap ___
visualize(heatmaps=explainability_map, image=image)