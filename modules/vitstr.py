'''
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
import torch.nn as nn
import logging
import torch.utils.model_zoo as model_zoo

from copy import deepcopy
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.factory import create_model

_logger = logging.getLogger(__name__)

__all__ = [
    'vitstr_tiny_patch16_224', 
    'vitstr_small_patch16_224', 
    'vitstr_base_patch16_224',
    'vitstr_tiny_distilled_patch16_224', 
    'vitstr_small_distilled_patch16_224',
    #'vitstr_base_distilled_patch16_224',
]

def create_vitstr(num_tokens, model=None, checkpoint_path=''):
    vitstr = create_model(
        model,
        pretrained=True,
        num_classes=num_tokens,
        drop_rate=0.,
        drop_connect_rate=None, 
        drop_path_rate=None,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        scriptable=False,
        checkpoint_path=checkpoint_path)

    # might need to run to get zero init head for transfer learning
    vitstr.reset_classifier(num_classes=num_tokens)

    return vitstr

class ViTSTR(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, seqlen=25):
        x = self.forward_features(x)
        x = x[:, :seqlen]

        # batch, seqlen, embsize
        b, s, e = x.size()
        x = x.reshape(b*s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    model.load_state_dict(state_dict, strict=strict)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@register_model
def vitstr_tiny_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = ViTSTR(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)

    model.default_cfg = _cfg(
            #url='https://github.com/roatienza/public/releases/download/v0.1-deit-tiny/deit_tiny_patch16_224-a1311bcf.pth'
            url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
    )

    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=_conv_filter)
    return model

@register_model
def vitstr_small_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = ViTSTR(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            #url="https://github.com/roatienza/public/releases/download/v0.1-deit-small/deit_small_patch16_224-cd65a155.pth"
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=checkpoint_filter_fn)
    return model

@register_model
def vitstr_base_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            #url='https://github.com/roatienza/public/releases/download/v0.1-deit-base/deit_base_patch16_224-b5f2ef4d.pth'
            url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=checkpoint_filter_fn)
    return model

@register_model
def vitstr_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = ViTSTR(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)

    model.default_cfg = _cfg(
            url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'
    )

    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=_conv_filter)
    return model


@register_model
def vitstr_small_distilled_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = ViTSTR(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=checkpoint_filter_fn)
    return model