
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

'''
    PIL resize (W,H)
    Torch resize is (H,W)
'''
class VGrid:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        line_width = np.random.randint(1, 4)
        image_stripe = np.random.randint(1, 4)
        n_lines = W // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            x = image_stripe*i + line_width*(i-1)
            draw.line([(x,0), (x,H)], width=line_width)

        return img

class HGrid:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        line_width = np.random.randint(1, 4)
        image_stripe = np.random.randint(1, 4)
        n_lines = H // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            y = image_stripe*i + line_width*(i-1)
            draw.line([(0,y), (W, y)], width=line_width)

        return img

class Grid:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        img = VGrid()(img)
        img = HGrid()(img)
        return img

