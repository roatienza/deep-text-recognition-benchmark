
import cv2
import numpy as np
from PIL import Image
from skimage import color
from .ops import plasma_fractal, clipped_zoom

'''
    PIL resize (W,H)
'''
class Fog:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        c = [(1.5, 2), (2., 2), (2.5, 1.7)]
        index = np.random.randint(0, len(c))
        c = c[index]

        img = np.array(img) / 255.
        max_val = img.max()
        img += c[0] * plasma_fractal(wibbledecay=c[1])[:W, :H][..., np.newaxis]
        img = np.clip(img * max_val / (max_val + c[0]), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class Frost:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7)]
        index = np.random.randint(0, len(c))
        c = c[index]

        filename = [resource_filename(__name__, 'frost/frost1.png'),
                    resource_filename(__name__, 'frost/frost2.png'),
                    resource_filename(__name__, 'frost/frost3.png'),
                    resource_filename(__name__, 'frost/frost4.jpg'),
                    resource_filename(__name__, 'frost/frost5.jpg'),
                    resource_filename(__name__, 'frost/frost6.jpg')]
        index = np.random.randint(0, len(filename))
        filename = filename[index]
        frost = cv2.imread(filename)
        #randomly crop and convert to rgb
        x_start, y_start = np.random.randint(0, frost.shape[0] - W), np.random.randint(0, frost.shape[1] - H)
        frost = frost[x_start:x_start + W, y_start:y_start + H][..., [2, 1, 0]]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.array(img)
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = np.clip(c[0] * np.array(img) + c[1] * frost, 0, 255)
        img = color.rgb2gray(img)

        return Image.fromarray(img.astype(np.uint8))


def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


