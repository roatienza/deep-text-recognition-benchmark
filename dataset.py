import os
import sys
import re
import six
import math
import lmdb
import torch
import cv2

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import auto_augment as augment

# fr https://github.com/kakaobrain/fast-autoaugment
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class NormalRotationTransform:
    def __init__(self, angle_std):
        self.angle_std = angle_std

    def __call__(self, x):
        # angle = torch.normal(mean=0., std=torch.Tensor([self.angle_std]))
        angle = np.random.normal(loc=0., scale=self.angle_std)
        return TF.rotate(x, angle)

class DataAugment(object):
    def __init__(self, opt):
        self.opt = opt
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.augment = augment.AutoAugment(dataset="imagenet")
        #self.lighting = augment.Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec'])

    def __call__(self, img):
        #img = transforms.Resize((self.opt.imgH, self.opt.imgW), interpolation=Image.BICUBIC)(img)
        img = img.resize((self.opt.imgH, self.opt.imgW), Image.BICUBIC)
        if self.opt.eval:
            img = transforms.ToTensor()(img)
            if self.opt.rgb and self.opt.auto_augment:
                img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])(img)
            return img

        #img.save("src.png" )

        if self.opt.auto_augment and self.opt.rgb:
            img = self.augment(img)
            img = transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    )(img)

        iswarp = np.random.uniform(0,1) < self.opt.warp_prob
        if self.opt.warp and iswarp:
            isflip = np.random.uniform(0,1) < 0.5
            if isflip:
                img = TF.vflip(img)

            img = np.array(img)
            W = self.opt.imgW
            H = self.opt.imgH
            W_25 = 0.25 * W
            W_50 = 0.50 * W
            W_75 = 0.75 * W
            r = np.random.uniform(0.9, 1.2)*H
            x1 = (r**2 - W_50**2)**0.5
            h1 = r - x1

            t = np.random.uniform(0.4,0.8)*H
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
            self.tps.estimateTransformation(np.array(dstpt).reshape((-1, N, 2)), np.array(srcpt).reshape((-1, N, 2)), matches)
            img = self.tps.warpImage(img)
            img = Image.fromarray(img)
            if isflip:
                img = TF.vflip(img)
            #img.save("curve.png" )

        isrotation = np.random.uniform(0,1) < self.opt.rotation_prob
        if self.opt.rotation and isrotation:
            angle = np.random.normal(loc=0., scale=self.opt.rotation_angle)
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = TF.rotate(img=img, angle=angle, resample=Image.BICUBIC, expand=True)
            img = transforms.Resize((self.opt.imgH, self.opt.imgW), interpolation=Image.BICUBIC)(img)
            #img.save("rotation.png" )

        isperspective = np.random.uniform(0,1) < self.opt.perspective_prob
        if self.opt.perspective and isperspective:
            # upper-left, upper-right, lower-left, lower-right
            src =  np.float32([[0, 0], [self.opt.imgW, 0], [0, self.opt.imgH], [self.opt.imgW, self.opt.imgH]])
            if np.random.uniform(0, 1) > 0.5:
                toprightY = np.random.uniform(0, 0.4)*self.opt.imgH
                bottomrightY = np.random.uniform(0.6, 1.0)*self.opt.imgH
                dest = np.float32([[0, 0], [self.opt.imgW, toprightY], [0, self.opt.imgH], [self.opt.imgW, bottomrightY]])
            else:
                topleftY = np.random.uniform(0, 0.4)*self.opt.imgH
                bottomleftY = np.random.uniform(0.6, 1.0)*self.opt.imgH
                dest = np.float32([[0, topleftY], [self.opt.imgW, 0], [0, bottomleftY], [self.opt.imgW, self.opt.imgH]])
            M = cv2.getPerspectiveTransform(src, dest)
            img = np.array(img)
            img = cv2.warpPerspective(img, M, (self.opt.imgW, self.opt.imgH) )
            #cv2.imwrite("perspective.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        img = transforms.ToTensor()(img)
        if self.opt.rgb and self.opt.auto_augment:
            #img = self.lighting(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])(img)


        #img.sub_(0.5).div_(0.5)

        #print(img.size())
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
        #    cv2.imwrite("dest.png" + name, img)
        #exit(0)

        return img


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


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


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.opt = opt

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = DataAugment(self.opt)
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        #else:
        #    transform = ResizeNormalize((self.imgW, self.imgH))
        #    image_tensors = [transform(image) for image in images]
        #    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
