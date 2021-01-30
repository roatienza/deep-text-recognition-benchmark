
import numpy as np
import skimage as sk
from PIL import Image

'''
    PIL resize (W,H)
'''
class GaussianNoise:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        #c = np.random.uniform(.08, .38)
        c = np.random.uniform(.08, .15)
        img = np.array(img) / 255.
        img = np.clip(img + np.random.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ShotNoise:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        #c = np.random.uniform(3, 60)
        c = np.random.uniform(3, 20)
        img = np.array(img) / 255.
        img = np.clip(np.random.poisson(img * c) / float(c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ImpulseNoise:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        #c = np.random.uniform(.03, .27)
        c = np.random.uniform(.03, .15)
        img = sk.util.random_noise(np.array(img) / 255., mode='s&p', amount=c) * 255
        return Image.fromarray(img.astype(np.uint8))


class SpeckleNoise:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        # c = np.random.uniform(.15, .6)
        c = np.random.uniform(.15, .3)
        img = np.array(img) / 255.
        img = np.clip(img + img * np.random.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))

