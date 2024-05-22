import math
import types
import torch
import sys
import inspect
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, InterpolationMode
import open_clip
from open_clip.transformer import VisionTransformer
from open_clip.timm_model import TimmModel
from einops import rearrange

from .utils_cogvlm import hooked_resblock_forward, \
    hooked_attention_forward, \
    hooked_resblock_timm_forward, \
    hooked_attentional_pooler_timm_forward, \
    vit_dynamic_size_forward, \
    min_max, \
    hooked_torch_multi_head_attention_forward


class LeWrapper(nn.Module):
    """
    Wrapper around OpenCLIP to add LeGrad to OpenCLIP's model while keeping all the functionalities of the original model.
    """

    # COPY MODEL
    def __init__(self, model, layer_index=-2):
        super(LeWrapper, self).__init__()
        # ------------ copy of model's attributes and methods ------------
        for attr in dir(model):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(model, attr))

        # ------------ activate hooks & gradient ------------
        self._activate_hooks(layer_index=layer_index)
        
            
    def _activate_hooks(self, layer_index):
        # ------------ identify model's type ------------
        print('Activating necessary hooks and gradients ....')
            
        if hasattr(self.model, 'vision'):
            
            # Store the original forward method for comparison
            original_forward = self.model.vision.forward
         
            # REMOVED DYNAMIC FORWARD
         
            # Set the custom dynamic size forward function 
            #self.model.vision.forward = types.MethodType(vit_dynamic_size_forward, self.model.vision)
            #self.visual.forward = types.MethodType(vit_dynamic_size_forward, self.visual)
            
            print("original forward: ")
            print(inspect.getsource(original_forward))
            # print(inspect.getsource(self.model.vision.forward ))
        
        total_layers = len(self.model.vision.transformer.layers)
        self.starting_depth = layer_index if layer_index >= 0 else total_layers + layer_index
        
        print("self.starting_depth: ", self.starting_depth)
        print('Hooks and gradients activated!')

        self._activate_self_attention_hooks()


    def _activate_self_attention_hooks(self):
        
        # Set gradients to TRUE for last X layers
        for name, param in self.model.vision.transformer.named_parameters():
            # print("paramater name", name)
            param.requires_grad = False
            if 'layers' in name:
                # get the depth from the parameter name
                depth = int(name.split('layers.')[1].split('.')[0])
                if depth >= self.starting_depth:
                    param.requires_grad = True
                    #print("changed param")
                    
        # Activate hooks
        for layer in range(self.starting_depth, len(self.model.vision.transformer.layers)):
            
            # TAKES ONLY THE TRANSFORMER BLOCKS (TRANSFORMER, MLP, LAYER NORM)
            current_layer = self.model.vision.transformer.layers[layer]
            
            # print("cogvlm transformer block attention forward", inspect.getsource(current_layer.attention.forward))
            
            # APPLY ATTENTION HOOK TO ATTENTION LAYER
            current_layer.attention.forward = types.MethodType(hooked_attention_forward, current_layer.attention)
            
            # APPLY FORWARD HOOK TO ENTIRE TRANSFORMER BLOCK
            # print("cogvlm transformer block forward", inspect.getsource(current_layer.forward))
            
            
            current_layer.forward = types.MethodType(hooked_resblock_forward, current_layer)
                        
    def compute_legrad(self, text_embedding, image=None, apply_correction=True):
        self.compute_legrad_cogvlm(text_embedding, image)

    def compute_legrad_cogvlm(self, text_embedding, image=None):
        
        print("shape text embedding: ", text_embedding.shape)
        num_prompts = text_embedding.shape[0]
        
        if image is not None:
            _ = self.model.vision(image)
            
                        
        blocks_list = list(dict(self.model.vision.transformer.layers.named_children()).values())
        

        image_features_list = []
    
        # Collect images features (activations) for specified layers/blocks and postprocess them
        for layer in range(self.starting_depth, len(self.visual.transformer.layers)):
            
            intermediate_feat = self.model.vision.transformer.layers[layer].feat_post_mlp
            print("intermediate_feat.shape: ", intermediate_feat.shape)
            
            # intermediate_feat = self.visual.ln_post(intermediate_feat.mean(dim=0)) @ self.visual.proj # FIND CORRECT PROJECTION
            
            # intermediate_feat = F.normalize(intermediate_feat, dim=-1)
            # image_features_list.append(intermediate_feat)
            
        
            
        # # Normalize features
        # num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
        # w = h = int(math.sqrt(num_tokens))
        
        # # SHOULD WORK FROM HERE
        # # ----- Get explainability map
        # accum_expl_map = 0
        # for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:], image_features_list)):
        #     self.visual.zero_grad()

        #     # Compute similarity between text and image features
        #     sim = text_embedding @ img_feat.transpose(-1, -2)  # [1, 1]
        #     one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_embedding.device)
        #     one_hot = torch.sum(one_hot * sim)
        #     attn_map = blocks_list[self.starting_depth + layer].attn.attention_map  # [b, num_heads, N, N]

        #     # -------- Get explainability map --------
            
        #     # Compute gradients
        #     grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[
        #         0]  # [batch_size * num_heads, N, N]
        #     grad = rearrange(grad, '(b h) n m -> b h n m', b=num_prompts)  # separate batch and attn heads
        #     grad = torch.clamp(grad, min=0.)

        #     # Average attention and reshape
        #     image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # average attn over [CLS] + patch tokens
            
        #     # Interpolate and normalize
        #     expl_map = rearrange(image_relevance, 'b (w h) -> 1 b w h', w=w, h=h)
        #     expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')  # [B, 1, H, W]
        #     accum_expl_map += expl_map

        # # Min-Max Norm
        # accum_expl_map = min_max(accum_expl_map)
        # return accum_expl_map
        
        
    
       



    # def compute_legrad_clip(self, text_embedding, image=None):
        
        
    #     num_prompts = text_embedding.shape[0]
        
    #     # Put the image through the encoder
    #     if image is not None:
    #         # image = image.repeat(num_prompts, 1, 1, 1)
    #         _ = self.encode_image(image)

    #     # Obtain all blocks
    #     blocks_list = list(dict(self.visual.transformer.resblocks.named_children()).values())
    #     image_features_list = []

    #     # Collect images features (activations) for specified layers/blocks and postprocess them
    #     for layer in range(self.starting_depth, len(self.visual.transformer.resblocks)):
    #         intermediate_feat = self.visual.transformer.resblocks[layer].feat_post_mlp  
    #         intermediate_feat = self.visual.ln_post(intermediate_feat.mean(dim=0)) @ self.visual.proj
    #         intermediate_feat = F.normalize(intermediate_feat, dim=-1)
    #         image_features_list.append(intermediate_feat)

    #     # Normalize features
    #     num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
    #     w = h = int(math.sqrt(num_tokens))
        
    #     # ----- Get explainability map
    #     accum_expl_map = 0
    #     for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:], image_features_list)):
    #         self.visual.zero_grad()

    #         # Compute similarity between text and image features
    #         sim = text_embedding @ img_feat.transpose(-1, -2)  # [1, 1]
    #         one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_embedding.device)
    #         one_hot = torch.sum(one_hot * sim)
    #         attn_map = blocks_list[self.starting_depth + layer].attn.attention_map  # [b, num_heads, N, N]

    #         # -------- Get explainability map --------
            
    #         # Compute gradients
    #         grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[
    #             0]  # [batch_size * num_heads, N, N]
    #         grad = rearrange(grad, '(b h) n m -> b h n m', b=num_prompts)  # separate batch and attn heads
    #         grad = torch.clamp(grad, min=0.)

    #         # Average attention and reshape
    #         image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # average attn over [CLS] + patch tokens
            
    #         # Interpolate and normalize
    #         expl_map = rearrange(image_relevance, 'b (w h) -> 1 b w h', w=w, h=h)
    #         expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')  # [B, 1, H, W]
    #         accum_expl_map += expl_map

    #     # Min-Max Norm
    #     accum_expl_map = min_max(accum_expl_map)
    #     return accum_expl_map

class LePreprocess(nn.Module):
    """
    Modify OpenCLIP preprocessing to accept arbitrary image size.
    """

    def __init__(self, preprocess, image_size):
        super(LePreprocess, self).__init__()
        self.transform = Compose(
            [
                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                preprocess.transforms[-3],
                preprocess.transforms[-2],
                preprocess.transforms[-1],
            ]
        )

    def forward(self, image):
        return self.transform(image)