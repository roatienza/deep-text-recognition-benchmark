'''
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
from timm.models.vision_transformer import VisionTransformer
from transform import data_augment
from utilities.misc import get_device, AverageMeter
from utilities.metrics import accuracy
from utilities.ui import progress_bar

import datetime
import os
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.nn as nn
import torchvision.utils

from timm.models import create_model, resume_checkpoint, load_checkpoint, convert_splitbn_model

def wordformer(num_tokens, model='vit_base_patch16_224'):
    model = create_model(
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
        checkpoint_path='')

    return model
