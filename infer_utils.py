import torch
import argparse
from PIL import Image
from torchvision import transforms


class ViTSTRFeatureExtractor:
    def __init__(self, input_channel=1, imgH=224, imgW=224):
        self.imgH = imgH
        self.imgW = imgW
        self.transform = NormalizePAD((input_channel, imgH, imgW))
    
    def __call__(self, img_path):
        img = Image.open(img_path).convert('L')
        img = img.resize((self.imgW, self.imgH), Image.BICUBIC)
        img = self.transform(img)
        img = torch.unsqueeze(img, dim=0)
        return img
    
class NormalizePAD:
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = max_size[2] // 2 
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return pad_img

class TokenLabelConverter:
    """ Convert between text-label and text-index """

    def __init__(self, args):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'
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


def get_args():
    parser = argparse.ArgumentParser(description='ViTSTR evaluation')

    parser.add_argument('--image', default=None, help='path to input image')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=224, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=224, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--input-channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--model', default="vitstr_small_patch16_224_aug_infer.pth", help='ViTSTR model')
    parser.add_argument('--gpu', action='store_true', help='use gpu for model inference')
    parser.add_argument('--time', action='store_true', help='perform speed benchmark')

    # For Raspberry Pi 4
    parser.add_argument('--quantized', action='store_true', help='Model quantization')
    parser.add_argument('--rpi', action='store_true', help='run on rpi 4')

    args = parser.parse_args()
    return args
