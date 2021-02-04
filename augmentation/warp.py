
import cv2
import numpy as np
from PIL import Image, ImageOps

'''
    PIL resize (W,H)
    Torch resize is (H,W)
'''
class Shrink:
    def __init__(self):
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        img = np.array(img)
        srcpt = list()
        dstpt = list()

        W_33 = 0.33 * W
        W_50 = 0.50 * W
        W_66 = 0.66 * W

        H_50 = 0.50 * H

        P = 0
        frac = 0.4

        # left-most
        srcpt.append([P, P])
        srcpt.append([P, H-P])
        x = np.random.uniform(0, frac)*W_33 
        y = np.random.uniform(0, frac)*H_50
        dstpt.append([P+x, P+y])
        dstpt.append([P+x, H-P-y])
        
        # 2nd left-most 
        srcpt.append([P+W_33, P])
        srcpt.append([P+W_33, H-P])
        dstpt.append([P+W_33, P+y])
        dstpt.append([P+W_33, H-P-y])
        
        # 3rd left-most 
        srcpt.append([P+W_66, P])
        srcpt.append([P+W_66, H-P])
        dstpt.append([P+W_66, P+y])
        dstpt.append([P+W_66, H-P-y])
        
        # right-most 
        srcpt.append([W-P, P])
        srcpt.append([W-P, H-P])
        dstpt.append([W-P-x, P+y])
        dstpt.append([W-P-x, H-P-y])

        N = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dstpt).reshape((-1, N, 2))
        src_shape = np.array(srcpt).reshape((-1, N, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        return img

class Stretch:
    def __init__(self):
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        img = np.array(img)
        srcpt = list()
        dstpt = list()

        W_33 = 0.33 * W
        W_50 = 0.50 * W
        W_66 = 0.66 * W

        H_50 = 0.50 * H

        P = 0
        frac = 0.4

        # left-most
        srcpt.append([P, P])
        srcpt.append([P, H-P])
        srcpt.append([P, H_50])
        x = np.random.uniform(0, frac)*W_33 if np.random.uniform(0,1) > 0.5 else 0
        dstpt.append([P+x, P])
        dstpt.append([P+x, H-P])
        dstpt.append([P+x, H_50])
        
        # 2nd left-most 
        srcpt.append([P+W_33, P])
        srcpt.append([P+W_33, H-P])
        x = np.random.uniform(-frac, frac)*W_33
        dstpt.append([P+W_33+x, P])
        dstpt.append([P+W_33+x, H-P])
        
        # 3rd left-most 
        srcpt.append([P+W_66, P])
        srcpt.append([P+W_66, H-P])
        x = np.random.uniform(-frac, frac)*W_33
        dstpt.append([P+W_66+x, P])
        dstpt.append([P+W_66+x, H-P])
        
        # right-most 
        srcpt.append([W-P, P])
        srcpt.append([W-P, H-P])
        srcpt.append([W-P, H_50])
        x = np.random.uniform(-frac, 0)*W_33 if np.random.uniform(0,1) > 0.5 else 0
        dstpt.append([W-P+x, P])
        dstpt.append([W-P+x, H-P])
        dstpt.append([W-P+x, H_50])

        N = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dstpt).reshape((-1, N, 2))
        src_shape = np.array(srcpt).reshape((-1, N, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        return img


class Distort:
    def __init__(self):
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        img = np.array(img)
        srcpt = list()
        dstpt = list()

        W_33 = 0.33 * W
        W_50 = 0.50 * W
        W_66 = 0.66 * W

        H_50 = 0.50 * H

        P = 0
        frac = 0.4

        # top pts
        srcpt.append([P, P])
        x = np.random.uniform(0, frac)*W_33
        y = np.random.uniform(0, frac)*H_50
        dstpt.append([P+x, P+y])
        
        srcpt.append([P+W_33, P])
        x = np.random.uniform(-frac, frac)*W_33
        y = np.random.uniform(0, frac)*H_50
        dstpt.append([P+W_33+x, P+y])
        
        srcpt.append([P+W_66, P])
        x = np.random.uniform(-frac, frac)*W_33
        y = np.random.uniform(0, frac)*H_50
        dstpt.append([P+W_66+x, P+y])
        
        srcpt.append([W-P, P])
        x = np.random.uniform(-frac, 0)*W_33
        y = np.random.uniform(0, frac)*H_50
        dstpt.append([W-P+x, P+y])

        # bottom pts
        srcpt.append([P, H-P])
        x = np.random.uniform(0, frac)*W_33
        y = np.random.uniform(-frac, 0)*H_50
        dstpt.append([P+x, H-P+y])
        
        srcpt.append([P+W_33, H-P])
        x = np.random.uniform(-frac, frac)*W_33
        y = np.random.uniform(-frac, 0)*H_50
        dstpt.append([P+W_33+x, H-P+y])
        
        srcpt.append([P+W_66, H-P])
        x = np.random.uniform(-frac, frac)*W_33
        y = np.random.uniform(-frac, 0)*H_50
        dstpt.append([P+W_66+x, H-P+y])
        
        srcpt.append([W-P, H-P])
        x = np.random.uniform(-frac, 0)*W_33
        y = np.random.uniform(-frac, 0)*H_50
        dstpt.append([W-P+x, H-P+y])

        N = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dstpt).reshape((-1, N, 2))
        src_shape = np.array(srcpt).reshape((-1, N, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        return img


class Curve:
    def __init__(self, square_side=224):
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.side = square_side

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size

        if H!=self.side or W!=self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        isflip = np.random.uniform(0,1) > 0.5
        if isflip:
            img = ImageOps.flip(img)
            #img = TF.vflip(img)

        img = np.array(img)
        w = self.side
        h = self.side
        w_25 = 0.25 * w
        w_50 = 0.50 * w
        w_75 = 0.75 * w
        r = np.random.uniform(0.8, 1.2)*h
        x1 = (r**2 - w_50**2)**0.5
        h1 = r - x1

        t = np.random.uniform(0.4,0.5)*h
        w2 = w_50*t/r
        hi = x1*t/r
        h2 = h1 + hi  

        sinb_2 = ((1 - x1/r)/2)**0.5
        cosb_2 = ((1 + x1/r)/2)**0.5
        w3 = w_50 - r*sinb_2
        h3 = r - r*cosb_2

        w4 = w_50 - (r-t)*sinb_2
        h4 = r - (r-t)*cosb_2

        w5 = 0.5*w2
        h5 = h1 + 0.5*hi
        h_50 = 0.50*h

        srcpt = [(0,0 ), (w,0 ), (w_50,0), (0,h  ), (w,h    ), (w_25,0), (w_75,0 ),  (w_50,h), (w_25,h), (w_75,h ), (0,h_50), (w,h_50 )]
        dstpt = [(0,h1), (w,h1), (w_50,0), (w2,h2), (w-w2,h2), (w3, h3), (w-w3,h3),  (w_50,t), (w4,h4 ), (w-w4,h4), (w5,h5 ), (w-w5,h5)]

        N = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dstpt).reshape((-1, N, 2))
        src_shape = np.array(srcpt).reshape((-1, N, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        if isflip:
            #img = TF.vflip(img)
            img = ImageOps.flip(img)
            rect = (0, self.side//2, self.side, self.side)
        else:
            rect = (0, 0, self.side, self.side//2)

        img = img.crop(rect)
        img = img.resize((W, H), Image.BICUBIC)
        return img


class Rotate:
    def __init__(self, square_side=224):
        self.side = square_side

    def __call__(self, img, iscurve=False, rotate_angle=30., prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size

        if H!=self.side or W!=self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        angle = np.random.normal(loc=0., scale=rotate_angle)
        angle = min(angle, 2*rotate_angle)
        angle = max(angle, -2*rotate_angle)
        expand = False if iscurve else True
        img = img.rotate(angle=angle, resample=Image.BICUBIC, expand=expand)
        img = img.resize((W, H), Image.BICUBIC)

        return img

class Perspective:
    def __init__(self):
        pass

    def __call__(self, img, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size

        # upper-left, upper-right, lower-left, lower-right
        src =  np.float32([[0, 0], [W, 0], [0, H], [W, H]])
        low = 0.3 
        high = 1 - low
        if np.random.uniform(0, 1) > 0.5:
            toprightY = np.random.uniform(0, low)*H
            bottomrightY = np.random.uniform(high, 1.0)*H
            dest = np.float32([[0, 0], [W, toprightY], [0, H], [W, bottomrightY]])
        else:
            topleftY = np.random.uniform(0, low)*H
            bottomleftY = np.random.uniform(high, 1.0)*H
            dest = np.float32([[0, topleftY], [W, 0], [0, bottomleftY], [W, H]])
        M = cv2.getPerspectiveTransform(src, dest)
        img = np.array(img)
        img = cv2.warpPerspective(img, M, (W, H) )
        img = Image.fromarray(img)

        return img

