import torch
import numpy as np
import argparse
import math
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TokenLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, args):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'
        #self.MASK = '[MASK]'

        #self.list_token = [self.GO, self.SPACE, self.MASK]
        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(args.character)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = args.batch_max_length + len(self.list_token)

    def encode(self, text):
        """ convert text-label into text-index.
        """
        length = [len(s) + len(self.list_token) for s in text]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.to(device)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


def get_device(verbose=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if verbose:
        print("Device:", device)
    return device

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


def get_args():
    parser = argparse.ArgumentParser(description='STR')

    parser.add_argument('--image', required=True, help='path to input image')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=224, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=224, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--input-channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    choices = ["vitstr_small_patch16_224_aug.pth"]
    parser.add_argument('--model', default=choices[0], help='Which vit/deit transformer model', choices=choices)

    args = parser.parse_args()
    return args
