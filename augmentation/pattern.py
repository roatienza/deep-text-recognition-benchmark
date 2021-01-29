
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
        line_width = np.random.randint(1, 3)
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
        line_width = np.random.randint(1, 3)
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

class RectGrid:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        #side = 224
        #img = img.resize((side, side), Image.BICUBIC)
        line_width = 1 #np.random.randint(1, 3) 
        image_stripe = 1 #np.random.randint(1, 3) 
        n_lines = ((H//2) // (line_width + image_stripe)) + 1
        draw = ImageDraw.Draw(img)
        x_center = W // 2
        y_center = H // 2
        for i in range(1, n_lines):
            dx = image_stripe*i + line_width*(i-1)
            dy = image_stripe*i + line_width*(i-1)
            x1 = x_center - (dx * W//H)
            y1 = y_center - dy
            x2 = x_center + (dx * W/H) 
            y2 = y_center + dy
            draw.rectangle([(x1,y1), (x2, y2)], width=line_width)

        #img = img.resize((W, H), Image.BICUBIC)
        return img
