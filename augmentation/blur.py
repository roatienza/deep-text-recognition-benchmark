
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

'''
    PIL resize (W,H)
'''
class GaussianBlur:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        kernel = [(27, 27), (29, 29), (31, 31)]
        index = np.random.randint(0, len(kernel))
        return transforms.GaussianBlur(kernel[index])(img)


class DefocusBlur:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        #c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
        c = [(3, 0.1), (4, 0.5), (6, 0.5)]
        index = np.random.randint(0, len(c))
        c = c[index]

        img = np.array(img) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
            n_channels = 3
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(n_channels):
            channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
        
        if isgray:
            img = img[:,:,0]
            img = np.squeeze(img)

        img = np.clip(channels, 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]

