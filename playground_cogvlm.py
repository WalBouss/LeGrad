import requests
from PIL import Image
import torch
import timm
import inspect
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from legrad import LeWrapper, LePreprocess, visualize

# def _get_text_embedding(model, tokenizer, query, device, image):
    
#     # # Prepare inputs using the custom build_conversation_input_ids method
#     # inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    
#     history = []
    
    
#     print("query: ", query)
#     print("tokenizer: ", tokenizer)
#     print("history: ", history)
#     print("images: ", image)
        
#     input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
    
#     # if image is None:
#     #     inputs = model.build_conversation_input_ids(tokenizer, query=query, history=history, template_version='base')
#     # else:
#     #     inputs = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
    
#     inputs = {
#             'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
#             'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
#             'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
#             'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
#         }
        

#     if 'cross_images' in inputs and inputs['cross_images']:
#             inputs['cross_images'] = [[inputs['cross_images'][0].to(DEVICE).to(torch.bfloat16)]]

#     gen_kwargs = {"max_length": 2048, "do_sample": False}

#     with torch.no_grad():
#         outputs = model.generate(**inputs, **gen_kwargs)
#         outputs = outputs[:, inputs['input_ids'].shape[1]:]
        
#     return outputs



# Define the preprocessing steps
# def create_cogvlm_preprocess(image_size=448):
#     from torchvision import transforms
#     preprocess_pipeline = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return preprocess_pipeline

# preprocess_pipeline = create_cogvlm_preprocess()



def _get_text_embedding(model, tokenizer, query, device, image):
    # Prepare inputs using the custom build_conversation_input_ids method
    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    
    # try:
    #     source_code = inspect.getsource(model.build_conversation_input_ids)
    #     print("Source Code:\n", source_code)
    # except TypeError:
        
    #     print("Couldn't retrieve source code. Function may be built-in or compiled.")
    
    
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
        'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
    }
    
    
        
    if inputs['images'] is None or not inputs['images'][0]:
        raise ValueError("The image input is not properly initialized or is None")


    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        text_embedding = model.generate(**inputs, **gen_kwargs)
        print("text_embedding shape: ",text_embedding.shape)
        
        # outputs = outputs[:, inputs['input_ids'].shape[1]:]
        
        # print("outputs shape: ", outputs.shape)
        # text_embeddings = tokenizer.decode(outputs[0])
        # print("text embeddings: ", text_embeddings)

    # print("inputs images len:", len(inputs['images']))
    # print("inputs images shape: ", inputs['images'][0][0].shape)
    
    processed_image = inputs['images'][0][0]
    return text_embedding, processed_image

def apply_transforms(image): 
    
    # Define the image size expected by the model
    image_size = 224  # This size should be confirmed from the model specifications

    resize_transform = Resize((image_size, image_size))
    to_tensor_transform = ToTensor()
    normalize_transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform = Compose([
    resize_transform,   
    to_tensor_transform,
    normalize_transform
    ])
    
    processed_image = transform(image)

    return processed_image
    

# ------- model's paramters -------
MODEL_PATH = "THUDM/cogvlm-chat-hf"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch_type = torch.bfloat16

# Obtain COGVLM model from HF
print("Loading Model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_type,
    low_cpu_mem_usage=True,
    load_in_4bit=None,
    trust_remote_code=True
).to(DEVICE).eval()

tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

# PROCESS IMAGE
image_path = '/home/jwiers/POPE/data/val2014/COCO_val2014_000000000042.jpg' 
image = Image.open(image_path).convert('RGB')

# image_tensor = preprocess_pipeline(image).unsqueeze(0).to(DEVICE)
text_emb, processed_image = _get_text_embedding(model, tokenizer, "a photo of a cat", DEVICE, image)
processed_image = processed_image.unsqueeze(0)

print("obtained output embedding")

print("text embedding shape: ", text_emb.shape)
print("processed image shape: ", processed_image.shape)

model = LeWrapper(model)

explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=text_emb)


# data_config = timm.data.resolve_model_data_config(model)




# # processed_image = apply_transforms(image)
# # processed_image = processed_image.unsqueeze(0).to(dtype=torch.bfloat16).to(DEVICE)
# print("processed_image shape: ", processed_image.shape)


# #explainability_map = model.compute_legrad_vmap_clip(image=image, text_embedding=text_embedding)

# # ___ (Optional): Visualize overlay of the image + heatmap ___
# visualize(heatmaps=explainability_map, image=image, save_path='/home/jwiers/VLM-Grounding/LeGrad/outputs/first_image.png')