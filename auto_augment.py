import random
import numpy as np
import PIL
import torch
import math
from PIL import Image, ImageEnhance, ImageOps

# AutoAugment Policies: https://arxiv.org/pdf/1805.09501.pdf
policies = { "cifar10" : 
             [
                [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
                [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
                [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
                [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
                [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)],
                [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],
                [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
                [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
                [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
                [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)],
                [('Color', 0.7, 7), ('TranslateX', 0.5, 8)],
                [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
                [('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)],
                [('Brightness', 0.9, 6), ('Color', 0.2, 6)],
                [('Solarize', 0.5, 2), ('Invert', 0.0, 3)],
                [('Equalize', 0.2, 0), ('AutoContrast', 0.6, 0)],
                [('Equalize', 0.2, 8), ('Equalize', 0.6, 4)],
                [('Color', 0.9, 9), ('Equalize', 0.6, 6)],
                [('AutoContrast', 0.8, 4), ('Solarize', 0.2, 8)],
                [('Brightness', 0.1, 3), ('Color', 0.7, 0)],
                [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
                [('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)],
                [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
                [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
                [('TranslateY', 0.7, 9), ('AutoContrast', 0.9, 1)],
             ],
             "svhn" :
             [
                 [('ShearX', 0.9, 4), ('Invert', 0.2, 3)],
                 [('ShearY', 0.9, 8), ('Invert', 0.7, 5)],
                 [('Equalize', 0.6, 5), ('Solarize', 0.6, 6)],
                 [('Invert', 0.9, 3), ('Equalize', 0.6, 3)],
                 [('Equalize', 0.6, 1), ('Rotate', 0.9, 3)],
                 [('ShearX', 0.9, 4), ('AutoContrast', 0.8, 3)],
                 [('ShearY', 0.9, 8), ('Invert', 0.4, 5)],
                 [('ShearY', 0.9, 5), ('Solarize', 0.2, 6)],
                 [('Invert', 0.9, 6), ('AutoContrast', 0.8, 1)],
                 [('Equalize', 0.6, 3), ('Rotate', 0.9, 3)],
                 [('ShearX', 0.9, 4), ('Solarize', 0.3, 3)],
                 [('ShearY', 0.8, 8), ('Invert', 0.7, 4)],
                 [('Equalize', 0.9, 5), ('TranslateY', 0.6, 6)],
                 [('Invert', 0.9, 4), ('Equalize', 0.6, 7)],
                 [('Contrast', 0.3, 3), ('Rotate', 0.8, 4)],
                 [('Invert', 0.8, 5), ('TranslateY', 0.0, 2)],
                 [('ShearY', 0.7, 6), ('Solarize', 0.4, 8)],
                 [('Invert', 0.6, 4), ('Rotate', 0.8, 4)],
                 [('ShearY', 0.3, 7), ('TranslateX', 0.9, 3)],
                 [('ShearX', 0.1, 6), ('Invert', 0.6, 5)],
                 [('Solarize', 0.7, 2), ('TranslateY', 0.6, 7)],
                 [('ShearY', 0.8, 4), ('Invert', 0.8, 8)],
                 [('ShearX', 0.7, 9), ('TranslateY', 0.8, 3)],
                 [('ShearY', 0.8, 5), ('AutoContrast', 0.7, 3)],
                 [('ShearX', 0.7, 2), ('Invert',0.1,5)],
             ],
             "imagenet" :
             [
                 [('Posterize', 0.4 ,8), ('Rotate', 0.6, 9)],
                 [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
                 [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
                 [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
                 [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
                 [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
                 [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
                 [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
                 [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
                 [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
                 [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
                 [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
                 [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
                 [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
                 [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
                 [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
                 [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
                 [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
                 [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
                 [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
                 [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
                 [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
                 [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
                 [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
                 [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
             ],
             "imagenet_no_rotation" :
             [
                 [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
                 [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
                 [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
                 [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
                 [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
                 [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
                 [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
                 [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
                 [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
                 [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
                 [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
                 [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
                 [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
                 [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
                 [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
                 [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
                 [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
                 [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
                 [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
             ],
          }


class AutoAugment:
    def __init__(self, dataset="cifar10"):
        if dataset == "cifar100":
            dataset = "cifar10"
        elif dataset == "svhn-core":
            dataset = "svhn"
        self.policies = policies[dataset]


    def __call__(self, img):
        img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img


operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'TranslateXAbs': lambda img, magnitude: translate_x_abs(img, magnitude),
    'TranslateYAbs': lambda img, magnitude: translate_y_abs(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'SolarizeAdd': lambda img, magnitude: solarize_add(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Posterize2': lambda img, magnitude: posterize2(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
}


def apply_policy(img, policy):
    for i in range(2):
        p = policy[i]
        if random.random() < p[1]:
            img = operations[p[0]](img, p[2])

    return img


def level(low, high, magnitude, random_mirror=True, asint=False):
    v = (magnitude / 10.) * (high - low) + low
    if random_mirror and random.random() > 0.5:
        v = -v
    v = int(v) if asint else v
    return v


def shear_level(magnitude):
    low = 0.
    high = 0.3
    return level(low, high, magnitude)


def shear_x(img, magnitude):
    v = shear_level(magnitude)
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def shear_y(img, magnitude):
    v = shear_level(magnitude)
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def translate_level(magnitude):
    low = 0.
    high = 0.45
    return level(low, high, magnitude)


def translate_level_abs(magnitude):
    low = 0.
    high = 10.
    return level(low, high, magnitude)


def translate_x(img, magnitude):
    v = translate_level(magnitude)
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def translate_x_abs(img, magnitude):
    v = translate_level_abs(magnitude)
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def translate_y(img, magnitude):
    v = translate_level(magnitude)
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def translate_y_abs(img, magnitude):
    v = translate_level_abs(magnitude)
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def rotate_level(magnitude):
    low =  0.
    high = 30.
    return level(low, high, magnitude)


def rotate(img, magnitude):
    v = rotate_level(magnitude)
    return img.rotate(v)


# PIL
def auto_contrast(img, magnitude):
    img = ImageOps.autocontrast(img)
    return img

# PIL 
def invert(img, magnitude):
    img = ImageOps.invert(img)
    return img

# PIL
def equalize(img, magnitude):
    img = ImageOps.equalize(img)
    return img

# PIL
def solarize(img, magnitude):
    magnitude = level(0, 256, magnitude, random_mirror=False, asint=False)
    img = ImageOps.solarize(img, magnitude)
    return img

def solarize_add(img, magnitude):
    threshold = 128
    image = np.array(img)
    added_image = image.astype(np.int64) + magnitude
    added_image = np.clip(added_image, 0, 255).astype(np.uint8)
    return Image.fromarray(np.where(image < threshold, added_image, image))


# PIL
def posterize(img, magnitude):
    magnitude = level(4, 8, magnitude, random_mirror=False, asint=True)
    img = ImageOps.posterize(img, magnitude)
    return img

def posterize2(img, magnitude):
    magnitude = level(0, 4, magnitude, random_mirror=False, asint=True)
    img = ImageOps.posterize(img, magnitude)
    return img

# PIL
def contrast(img, magnitude):
    magnitude = level(0.1, 1.9, magnitude, random_mirror=False, asint=False)
    img = ImageEnhance.Contrast(img).enhance(magnitude)
    return img


# PIL
def color(img, magnitude):
    magnitude = level(0.1, 1.9, magnitude, random_mirror=False, asint=False)
    img = ImageEnhance.Color(img).enhance(magnitude)
    return img


# PIL
def brightness(img, magnitude):
    magnitude = level(0.1, 1.9, magnitude, random_mirror=False, asint=False)
    img = ImageEnhance.Brightness(img).enhance(magnitude)
    return img


# PIL
def sharpness(img, magnitude):
    magnitude = level(0.1, 1.9, magnitude, random_mirror=False, asint=False)
    img = ImageEnhance.Sharpness(img).enhance(magnitude)
    return img


# CutOut from: https://github.com/uoguelph-mlrg/Cutout
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class Cutout_PIL:
    """Randomly mask out one or more patches from an image.

    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length=16, n_holes=1):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (H, W, C).
        Returns:
            Tensor: Image with length x length cut out of it.
        """
        img = np.array(img)
        h = img.shape[0]
        w = img.shape[1]

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            img[y1: y2, x1: x2, :] = 0.

        img = Image.fromarray(img)

        return img

# From FastAutoAugment: https://github.com/kakaobrain/fast-autoaugment
class EfficientNetRandomCrop:
    def __init__(self,
                 imgsize, 
                 min_covered=0.1, 
                 aspect_ratio_range=(3./4, 4./3), 
                 area_range=(0.08, 1.0), 
                 max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCrop(imgsize)

    def __call__(self, img):
        # https://github.com/tensorflow/tensorflow/blob/
        # 9274bcebb31322370139467039034f8ff852b004/
        # tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L111
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img)      
                # https://github.com/tensorflow/tpu/blob/master/models/
                # official/efficientnet/preprocessing.py#L102

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            return img.crop((x, y, x + width, y + height))

        return self._fallback(img)


class EfficientNetCenterCrop:
    def __init__(self, imgsize):
        self.imgsize = imgsize

    def __call__(self, img):
        """Crop the given PIL Image and resize it to desired size.

        Args:
            img (PIL Image): Image to be cropped. 
                (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

class Lighting:
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

# fr https://github.com/clovaai/CutMix-PyTorch
def cutmix(x, target, beta, device):
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(x.size()[0]).to(device)
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, target_a, target_b, lam


# fr https://github.com/clovaai/CutMix-PyTorch
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# fr https://github.com/facebookresearch/mixup-cifar10
def mixup(x, target, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    x = lam * x + (1 - lam) * x[index, :]
    target_a, target_b = target, target[index]
    return x, target_a, target_b, lam

