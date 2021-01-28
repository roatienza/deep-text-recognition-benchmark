import os
import sys
import re
import six
import math
import lmdb
import torch
import cv2

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import auto_augment as augment

class Warp:
    def __init__(self, opt):
        self.opt = opt
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, img):
        '''
            PIL resize (W,H)
            Torch resize is (H,W)
        '''

        side = 224
        if self.opt.imgH!=side and self.opt.imgW!=side:
            img = img.resize((side, side), Image.BICUBIC)

        if iswarp:
        isflip = np.random.uniform(0,1) < 0.5
        if isflip:
            img = TF.vflip(img)

        img = np.array(img)
        W = side
        H = side
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
            rect = (0, side//2, side, side)
        else:
            rect = (0, 0, side, side//2)

        img = img.crop(rect)
        #img.save("curve.png" )
        img = img.resize((side, side), Image.BICUBIC)

        if isrotation:
            angle = np.random.normal(loc=0., scale=self.opt.rotation_angle)
            angle = min(angle, self.opt.rotation_angle)
            angle = max(angle, -self.opt.rotation_angle)
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            expand = True
            if iswarp:
                expand = False
            img = TF.rotate(img=img, angle=angle, resample=Image.BICUBIC, expand=expand)
            #img.save("rotation.png" )

        if isperspective and not isrotation:
            # upper-left, upper-right, lower-left, lower-right
            src =  np.float32([[0, 0], [side, 0], [0, side], [side, side]])
            low = 0.3 if iswarp else 0.4
            high = 1 - low
            if np.random.uniform(0, 1) > 0.5:
                toprightY = np.random.uniform(0, low)*side
                bottomrightY = np.random.uniform(high, 1.0)*side
                dest = np.float32([[0, 0], [side, toprightY], [0, side], [side, bottomrightY]])
            else:
                topleftY = np.random.uniform(0, low)*side
                bottomleftY = np.random.uniform(high, 1.0)*side
                dest = np.float32([[0, topleftY], [side, 0], [0, bottomleftY], [side, side]])
            M = cv2.getPerspectiveTransform(src, dest)
            img = np.array(img)
            img = cv2.warpPerspective(img, M, (side, side) )
            #cv2.imwrite("perspective.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            img = Image.fromarray(img)

        #img = img.resize((self.opt.imgH, self.opt.imgW), Image.BICUBIC)

        # PIL size is (W,H) while Transforms (H, W)
        if (self.opt.imgW, self.opt.imgH) != img.size:
            img = transforms.Resize((self.opt.imgH, self.opt.imgW), interpolation=Image.BICUBIC)(img)

        img = transforms.ToTensor()(img)

        #if self.opt.rgb and self.opt.auto_augment:
        #    if self.opt.lighting:
        #        img = self.lighting(img)
        #    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                               std=[0.229, 0.224, 0.225])(img)

        if self.scale:
            img.sub_(0.5).div_(0.5)

        #if self.opt.rgb:
        #    img = img.permute(1,2,0)
        #else:
        #    img = img[0].squeeze()
        #img = img.cpu().numpy()
        #img = (((img + 1) * 0.5) * 255).astype(np.uint8)
        #if self.opt.rgb:
        #    cv2.imwrite("dest.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #else:
        #    img = np.expand_dims(img, axis=2)
        #    img = np.repeat(img, 3, axis=2)
        #    cv2.imwrite("dest-gray.png", img)
        #exit(0)

        return img

