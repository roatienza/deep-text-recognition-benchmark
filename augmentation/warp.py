import cv2

import numpy as np
import torch
import torchvision.transforms.functional as TF

from PIL import Image

class Curve:
    def __init__(self, opt):
        self.opt = opt
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.side = 224

    def __call__(self, img):
        '''
            PIL resize (W,H)
            Torch resize is (H,W)
        '''

        if self.opt.imgH!=self.side and self.opt.imgW!=self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        isflip = np.random.uniform(0,1) < 0.5
        if isflip:
            img = TF.vflip(img)

        img = np.array(img)
        W = self.side
        H = self.side
        W_25 = 0.25 * W
        W_50 = 0.50 * W
        W_75 = 0.75 * W
        r = np.random.uniform(0.8, 1.2)*H
        x1 = (r**2 - W_50**2)**0.5
        h1 = r - x1

        t = np.random.uniform(0.4,0.5)*H
        w2 = W_50*t/r
        hi = x1*t/r
        h2 = h1 + hi  

        sinb_2 = ((1 - x1/r)/2)**0.5
        cosb_2 = ((1 + x1/r)/2)**0.5
        w3 = W_50 - r*sinb_2
        h3 = r - r*cosb_2

        w4 = W_50 - (r-t)*sinb_2
        h4 = r - (r-t)*cosb_2

        w5 = 0.5*w2
        h5 = h1 + 0.5*hi
        H_50 = 0.50*H

        srcpt = [(0,0 ), (W,0 ), (W_50,0), (0,H  ), (W,H    ), (W_25,0), (W_75,0 ),  (W_50,H), (W_25,H), (W_75,H ), (0,H_50), (W,H_50 )]
        dstpt = [(0,h1), (W,h1), (W_50,0), (w2,h2), (W-w2,h2), (w3, h3), (W-w3,h3),  (W_50,t), (w4,h4 ), (W-w4,h4), (w5,h5 ), (W-w5,h5)]

        N = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dstpt).reshape((-1, N, 2))
        src_shape = np.array(srcpt).reshape((-1, N, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        if isflip:
            img = TF.vflip(img)
            rect = (0, self.side//2, self.side, self.side)
        else:
            rect = (0, 0, self.side, self.side//2)

        img = img.crop(rect)
        #img.save("curve.png" )
        
        #img = img.resize((self.side, self.side), Image.BICUBIC)
        #if (self.opt.imgW, self.opt.imgH) != img.size:
        # img = transforms.Resize((self.opt.imgH, self.opt.imgW), interpolation=Image.BICUBIC)(img)
        img = img.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)
        return img


class Rotate:
    def __init__(self, opt):
        self.opt = opt
        self.side = 224

    def __call__(self, img, iswarp=False):
        '''
            PIL resize (W,H)
            Torch resize is (H,W)
        '''

        if self.opt.imgH!=self.side and self.opt.imgW!=self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        angle = np.random.normal(loc=0., scale=self.opt.rotate_angle)
        angle = min(angle, self.opt.rotate_angle)
        angle = max(angle, -self.opt.rotate_angle)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        expand = True
        if iswarp:
            expand = False
        img = TF.rotate(img=img, angle=angle, resample=Image.BICUBIC, expand=expand)
        #img.save("rotate.png" )
        img = img.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)

        return img

class Perspective:
    def __init__(self, opt):
        self.opt = opt
        self.side = 224

    def __call__(self, img, isrotate=False):
        '''
            PIL resize (W,H)
            Torch resize is (H,W)
        '''

        if self.opt.imgH!=self.side and self.opt.imgW!=self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        if not isrotate:
            # upper-left, upper-right, lower-left, lower-right
            src =  np.float32([[0, 0], [self.side, 0], [0, self.side], [self.side, self.side]])
            low = 0.3 
            high = 1 - low
            if np.random.uniform(0, 1) > 0.5:
                toprightY = np.random.uniform(0, low)*self.side
                bottomrightY = np.random.uniform(high, 1.0)*self.side
                dest = np.float32([[0, 0], [self.side, toprightY], [0, self.side], [self.side, bottomrightY]])
            else:
                topleftY = np.random.uniform(0, low)*self.side
                bottomleftY = np.random.uniform(high, 1.0)*self.side
                dest = np.float32([[0, topleftY], [self.side, 0], [0, bottomleftY], [self.side, self.side]])
            M = cv2.getPerspectiveTransform(src, dest)
            img = np.array(img)
            img = cv2.warpPerspective(img, M, (self.side, self.side) )
            #cv2.imwrite("perspective.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            img = Image.fromarray(img)

        img = img.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)

        return img

