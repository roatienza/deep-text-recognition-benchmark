
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from wand.image import Image as WandImage
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from io import BytesIO

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
        kernel = [(27, 27), (29, 29), (31,31)]
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

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

class MotionBlur:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        #c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]
        c = [(10, 3), (15, 5), (15, 8)]
        index = np.random.randint(0, len(c))
        c = c[index]

        output = BytesIO()
        img.save(output, format='PNG')
        img = MotionImage(blob=output.getvalue())

        img.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        img = cv2.imdecode(np.fromstring(img.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img.astype(np.uint8))

class GlassBlur:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        #c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3)]
        index = np.random.randint(0, len(c))
        c = c[index]

        img = np.uint8(gaussian(np.array(img) / 255., sigma=c[0], multichannel=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(H - c[1], c[1], -1):
                for w in range(W - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    img[h, w], img[h_prime, w_prime] = img[h_prime, w_prime], img[h, w]

        img = np.clip(gaussian(img / 255., sigma=c[0], multichannel=True), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ZoomBlur:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02)]
        index = np.random.randint(0, len(c))
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = (np.array(img) / 255.).astype(np.float32)
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        out = np.zeros_like(img)
        #for zoom_factor in c:
        #    out += clipped_zoom(img, zoom_factor)

        img = (img + out) / (len(c) + 1)

        if isgray:
            img = img[:,:,0]
            img = np.squeeze(img)

        img = np.clip(img, 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


def clipped_zoom(img, zoom_factor):
    h = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]

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

