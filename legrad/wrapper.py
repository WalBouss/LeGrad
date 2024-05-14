import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, InterpolationMode
import open_clip
from open_clip.transformer import VisionTransformer
from open_clip.timm_model import TimmModel
from einops import rearrange

from .utils import hooked_resblock_forward, \
    hooked_attention_forward, \
    hooked_resblock_timm_forward, \
    hooked_attentional_pooler_timm_forward, \
    vit_dynamic_size_forward, \
    min_max, \
    hooked_torch_multi_head_attention_forward

#


class LeWrapper(nn.Module):
    """
    Wrapper around CogVLM to add LeGrad while keeping all functionalities of the original model.
    """

    def __init__(self, model, layer_index=-2):
        super(LeWrapper, self).__init__()
        # Copy all attributes and methods from the original model
        for attr in dir(model):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(model, attr))

        # Activate hooks and gradient
        self._activate_hooks(layer_index=layer_index)

    def _activate_hooks(self, layer_index):
        # Identify model's type
        print('Activating necessary hooks and gradients ....')

        # Check for vision components and apply hooks
        if hasattr(self.model, 'vision'):
            self.visual = self.model.vision.transformer
            self.hooks = []
            for layer in self.visual.layers[layer_index:]:
                handle = layer.register_forward_hook(self.save_outputs_hook)
                self.hooks.append(handle)
        elif hasattr(self.model, 'cross_vision'):
            self.visual = self.model.cross_vision.vit.model.blocks
            self.hooks = []
            for layer in self.visual[layer_index:]:
                handle = layer.register_forward_hook(self.save_outputs_hook)
                self.hooks.append(handle)
        else:
            raise ValueError("Model currently not supported, or does not contain a Vision Transformer component.")

        print('Hooks and gradients activated!')

    def save_outputs_hook(self, module, input, output):
        self.outputs = output

    def compute_legrad(self, text_embedding, image=None, apply_correction=True):
        # Assuming a similar compute_legrad method for CogVLM
        if hasattr(self.model, 'vision') or hasattr(self.model, 'cross_vision'):
            return self.compute_legrad_cogvlm(text_embedding, image)

    def compute_legrad_cogvlm(self, text_embedding, image=None):
        # Custom implementation for CogVLM
        num_prompts = text_embedding.shape[0]
        if image is not None:
            _ = self.encode_image(image)

        # Example processing loop for CogVLM's Vision Transformer
        blocks_list = list(self.visual)  # Adjust based on actual structure
        image_features_list = []

        for layer in range(self.starting_depth, len(self.visual)):
            intermediate_feat = self.visual[layer].feat_post_mlp  # Adjust based on actual structure
            intermediate_feat = F.normalize(intermediate_feat, dim=-1)
            image_features_list.append(intermediate_feat)

        num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
        w = h = int(math.sqrt(num_tokens))

        # Get explainability map
        accum_expl_map = 0
        for layer, img_feat in enumerate(image_features_list):
            self.visual.zero_grad()
            sim = text_embedding @ img_feat.transpose(-1, -2)  # [1, 1]
            one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_embedding.device)
            one_hot = torch.sum(one_hot * sim)

            attn_map = blocks_list[self.starting_depth + layer].attn.attention_map  # Adjust based on actual structure

            # Get explainability map
            grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[0]
            grad = torch.clamp(grad, min=0.)
            image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # Adjust based on actual structure
            expl_map = rearrange(image_relevance, 'b (w h) -> 1 b w h', w=w, h=h)
            expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')  # [B, 1, H, W]
            accum_expl_map += expl_map

        accum_expl_map = (accum_expl_map - accum_expl_map.min()) / (accum_expl_map.max() - accum_expl_map.min())
        return accum_expl_map


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
