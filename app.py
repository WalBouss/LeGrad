import requests
import numpy as np
import cv2 as cv2
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip

import gradio as gr

from legrad import LeWrapper, LePreprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_index = -2  # will run on cpu
image_size = 448
# ---------- Init CLIP Model ----------
model_name = 'ViT-B-16'
pretrained = 'laion2b_s34b_b88k'
patch_size = 16

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
tokenizer = open_clip.get_tokenizer(model_name)

# ---------- Apply LeGrad's wrappers ----------
model = LeWrapper(model)
preprocess = LePreprocess(preprocess=preprocess, image_size=image_size)


# ---------- Function to load image from URL ----------
def change_to_url(url):
    img_pil = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    return img_pil


def _get_text_embedding(model, tokenizer, classes: list, device):
    prompts = [f'a photo of a {cls}.' for cls in classes]

    tokenized_prompts = tokenizer(prompts).to(device)

    text_embedding = model.encode_text(tokenized_prompts)
    text_embedding = F.normalize(text_embedding, dim=-1)
    return text_embedding.unsqueeze(0)

# ---------- Function to convert logits to heatmaps ----------
def logits_to_heatmaps(logits, image_cv):
    logits = logits[0, 0].detach().cpu().numpy()
    logits = (logits * 255).astype('uint8')
    heat_map = cv2.applyColorMap(logits, cv2.COLORMAP_JET)
    viz = 0.4 * image_cv + 0.6 * heat_map
    viz = cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB)
    return viz


# ---------- Main visualization function ----------
def viz_func(url, image, text_query):
    image_torch = preprocess(image).unsqueeze(0).to(device)
    text_emb = _get_text_embedding(model, tokenizer, classes=[text_query], device=device)

    # ------- Get LeGrad output -------
    logits_legrad = model.compute_legrad(image=image_torch, text_embedding=text_emb)
    # ------- Get Heatmpas -------
    image_cv = cv2.cvtColor(np.array(image.resize((image_size, image_size))), cv2.COLOR_RGB2BGR)

    viz_legrad = logits_to_heatmaps(logits=logits_legrad, image_cv=image_cv)
    return viz_legrad

inputs = [
    gr.Textbox(label="Paste the url to the  selected image"),
    gr.Image(type="pil", interactive=True, label='Select An Image'),
    gr.Textbox(label="Text query"),
    ]


with gr.Blocks(css="#gradio-app-title { text-align: center; }") as demo:
    gr.Markdown(
        """
        # **LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity**
        ### This demo that showcases LeGrad method to visualize the important regions in an image that correspond to a given text query.
        The model used is OpenCLIP-ViT-B-16 (weights: `laion2b_s34b_b88k`)
        """
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown('# Select An Image')
            selected_image = gr.Image(type="pil", interactive=True, label='')
            gr.Markdown('## Paste the url to the  selected image')
            url_query = gr.Textbox(label="")
            gr.Markdown('# Create your Own query')
            text_query = gr.Textbox(label='')
            run_button = gr.Button(icon='https://cdn-icons-png.flaticon.com/512/3348/3348036.png')

            inputs[0].change(fn=change_to_url, outputs=inputs[1], inputs=inputs[0])
            gr.Markdown('## LeGrad Explanation')
            le_grad_output = gr.Image(label='LeGrad')

            run_button.click(fn=viz_func,
                inputs=[url_query, selected_image, text_query],
                outputs=[le_grad_output])

        with gr.Column():
            gr.Markdown('# Select a Premade Example')
            gr.Examples(
                examples=[
                    ["gradio_app/assets/cats_remote_control.jpeg", "cat"],
                    ["gradio_app/assets/cats_remote_control.jpeg", "remote control"],
                    ["gradio_app/assets/la_baguette.webp", "la baguette"],
                    ["gradio_app/assets/la_baguette.webp", "beret"],
                    ["gradio_app/assets/pokemons.jpeg", "Pikachu"],
                    ["gradio_app/assets/pokemons.jpeg", "Bulbasaur"],
                    ["gradio_app/assets/pokemons.jpeg", "Charmander"],
                    ["gradio_app/assets/pokemons.jpeg", "Pokemons"],
                ],
                inputs=[selected_image, text_query],
                label=''
            )

demo.queue()
demo.launch()