import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math

class Grid:
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=0, prob=0.5):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        #h = img.shape[0]
        #w = img.shape[1]
        w, h = img.size
        #print("w: ", w)
        #print("h: ", h)
        #exit(0)
        
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image 
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        #hh = math.ceil((math.sqrt(h*h + w*w)))
        hh = (int)((h*h + w*w)**0.5) + 1
        
        d = np.random.randint(self.d1, self.d2)
        #d = self.d
        
        # maybe use ceil? but i guess no big difference
        #self.l = math.ceil(d*self.ratio)
        self.l = (int)(d*self.ratio)
        
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d//2, d)
        st_w = np.random.randint(d//2, d)
        for i in range(-1, hh//d+1):
                s = d*i + st_h
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
                s = d*i + st_w
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[:,s:t] *= 0
        #r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        #mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        #print(80 * "^")
        #print(mask.shape)
        #mask = torch.from_numpy(mask.copy()).float().cuda()
        #print(80 * "%")
        if self.mode == 1:
            mask = 1-mask
        
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img)
        print(80 * "*")
        print(img.shape)
        if img.shape[2] > 1: 
            mask = np.repeat(mask, img.shape[2], axis=2)
        exit(0)
        #mask = mask.expand_as(img)
        #print(mask.shape)
        img = img * mask 

        #print(80 * "*")
        #print(img.shape)
        #img = np.transpose(img, (1,2,0))
        #print(img.shape)
        #print(img.shape)
        img = Image.fromarray(img)
        #img.save("grid.png")
        #cv2.imwrite("grid.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return img

class GridMask:
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def __call__(self, x):
        return x
        #if not self.training:
        #    return x
        #n,c,h,w = x.size()
        #y = []
        #for i in range(n):
        #    y.append(self.grid(x[i]))
        #y = torch.cat(y).view(n,c,h,w)

        #return y

