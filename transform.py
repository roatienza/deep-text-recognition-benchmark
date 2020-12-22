'''
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision.transforms as transforms
from PIL import Image

# mean and std fr https://github.com/pytorch/examples/blob/master/imagenet/main.py
imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

# mean and std fr https://github.com/kakaobrain/fast-autoaugment
cifar_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010])

# mean and std fr https://github.com/uoguelph-mlrg/Cutout
svhn_normalize = transforms.Normalize(mean=[0.4309803921568628, 0.4301960784313726, 0.4462745098039216],
                                      std=[0.19647058823529412, 0.1984313725490196, 0.19921568627450978])

# fr https://github.com/kakaobrain/fast-autoaugment
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}


def data_augment(dataset="cifar10",
                 size=224,
                 length=16,
                 cutout=False,
                 auto_augment=False,
                 no_basic_augment=False):

    if dataset == "imagenet":
        normalize = imagenet_normalize
    elif dataset == "svhn":
        normalize = svhn_normalize
    else:
        normalize = cifar_normalize

    # SVHN on Wide ResNet has no pre-processing 
    # https://arxiv.org/pdf/1605.07146.pdf
    # dropout = 0.4 is used
    #if no_basic_augment or dataset == "svhn" or dataset == "svhn-core":
    if no_basic_augment:
        transform_train = []
        print("No basic augment")
    else:
        # imagenet baseline transform from: 
        # https://github.com/clovaai/CutMix-PyTorch
        if dataset == "imagenet":
            transform_train = [
                #augment.EfficientNetRandomCrop(input_size),
                #transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
            ]
            print("RandomCrop=224 + RandomHorizontalFlip")
        elif dataset == "svhn" or dataset == "svhn-core":
            transform_train = [
                    transforms.Resize(size),
            ]
            print("RandomCrop")
        else:
            transform_train = [
                    transforms.Resize(size),
                    transforms.RandomHorizontalFlip(),
            ]
            print("RandomCrop + RandomHorizontalFlip")

    if dataset == "imagenet":
        transform_train.extend([
            transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
            ),
            transforms.ToTensor(),
            augment.Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        print("Test Resize=256, CenterCrop=224")
        print("ImageNet Train and Test Transforms")
    else:
        transform_train.extend([
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ])
        print("CIFAR/SVHN Train and Test Transforms")

    # cutout comes after normalize
    # if before normalize, use CutOut_PIL
    if cutout:
        transform_train.append(augment.Cutout(length=length))
        print("CutOut: ", length)

    transform_train = transforms.Compose(transform_train)

    return {'train' : transform_train, 'test' : transform_test}


def color_jitter_transform():
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(
                    brightness=1.0,
                    contrast=1.0,
                    saturation=1.0,
            ),
            transforms.ToTensor(),
            imagenet_normalize,
        ])
    print("ImageNet Color Jitter")

    return {'train' : transform_test, 'test' : transform_test}

def random_resized_crop_transform():
    transform_test = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.ToTensor(),
            imagenet_normalize,
        ])
    print("ImageNet RandomResizedCrop")

    return {'train' : transform_test, 'test' : transform_test}
