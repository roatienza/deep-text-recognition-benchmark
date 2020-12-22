'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = 100
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val < self.min:
            self.min = val
        if val > self.max:
            self.max = val

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res


def get_device(verbose=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #if torch.cuda.device_count() > 1:
    #    print("Available GPUs:", torch.cuda.device_count())
    #    # model = nn.DataParallel(model)
    if verbose:
        print("Device:", device)
    return device


def get_args():
    parser = argparse.ArgumentParser(description='MIMax')
    parser.add_argument('--config',
                        default=None,
                        help='Load a pre-defined training configuration')
    parser.add_argument('--n-runs',
                        type=int,
                        default=1,
                        help='Number of runs for MIMax training')
    parser.add_argument('--best-top1',
                        type=float,
                        default=0,
                        metavar='N',
                        help='Best top 1 accuracy')
    parser.add_argument('--best-top5',
                        type=float,
                        default=0,
                        metavar='N',
                        help='Best top 5 accuracy')
    parser.add_argument('--best-model',
                        default=None,
                        help='Best Model')
    parser.add_argument('--adam',
                        default=False,
                        action='store_true',
                        help='Use Adam optimizer')
    parser.add_argument('--steplr',
                        default=False,
                        action='store_true',
                        help='Use Step LR')
    parser.add_argument('--multisteplr',
                        default=False,
                        action='store_true',
                        help='Use Multi Step LR')
    parser.add_argument('--plateau',
                        default=False,
                        action='store_true',
                        help='Use reduce on plataeu')
    parser.add_argument('--ce-weight',
                        type=float,
                        default=1.0,
                        metavar='N',
                        help='Cross-entropy weight on double classifier')
    parser.add_argument('--mi-weight',
                        type=float,
                        default=1.0,
                        metavar='N',
                        help='Entropy weight on double classifier')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        metavar='N',
                        help='Learning rate scheduler gamma')
    parser.add_argument('--step-size',
                        type=int,
                        default=400,
                        metavar='N',
                        help='Learning rate step size')
    parser.add_argument('--fixed-train',
                        default=False,
                        action='store_true',
                        help='Not both train samples are transformed')
    parser.add_argument('--dl',
                        default="l1",
                        help='MI divergence loss')
    parser.add_argument('--n-units',
                        type=int,
                        default=0,
                        metavar='N',
                        help='Number of units of 1st layer of Q network')
    parser.add_argument('--dl-weight',
                        type=float,
                        default=4,
                        metavar='N',
                        help='Divergence loss weight')
    #parser.add_argument('--n-heads',
    #                    type=int,
    #                    default=1,
    #                    metavar='N',
    #                    help='Number of heads')
    parser.add_argument('--pool-size',
                        type=int,
                        default=1,
                        metavar='N',
                        help='Average pooling size')
    parser.add_argument('--n-channels',
                        type=int,
                        default=None,
                        metavar='N',
                        help='Number of channels')
    parser.add_argument('--n-classes',
                        type=int,
                        default=None,
                        metavar='N',
                        help='Number of classes')
    #parser.add_argument('--head-index',
    #                    type=int,
    #                    default=0,
    #                    metavar='N',
    #                    help='Which encoder head to use')
    parser.add_argument('--init-backbone',
                        default=None,
                        action='store_true',
                        help='Initialize backbone')
    parser.add_argument('--init-extractor',
                        default=None,
                        action='store_true',
                        help='Initialize feature extractor')
    parser.add_argument('--weights-std',
                        type=float,
                        default=None,
                        metavar='N',
                        help='Linear layer initial weights std (0.01 for baseline, 0.2 for mimax)')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=None,
                        metavar='N',
                        help='Optimizer weight decay')
    parser.add_argument('--batch-size',
                        type=int,
                        default=None,
                        metavar='N',
                        help='Batch size for training MIMax')
    parser.add_argument('--epochs',
                        type=int,
                        default=None,
                        metavar='N',
                        help='Number of epochs of (270 ImageNet, 200 CIFAR, 160 SVHN)')
    parser.add_argument('--lr',
                        type=float,
                        default=None,
                        metavar='N',
                        help='Learning rate')
    parser.add_argument('--dropout',
                        type=float,
                        default=None,
                        metavar='N',
                        help='Dropout (when applicable)')
    parser.add_argument('--nesterov',
                        default=False,
                        action='store_true',
                        help='Use Nesterov momentum on SGD')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='N',
                        help='SGD momentum (when applicable)')
    parser.add_argument('--auto-augment',
                        default=False,
                        action='store_true',
                        help='Use automatic data augmentation')
    parser.add_argument('--cutout',
                        default=False,
                        action='store_true',
                        help='Use cut out data augmentation')
    parser.add_argument('--no-basic-augment',
                        default=None,
                        action='store_true',
                        help='No basic data augmentation (crop, flip)')
    parser.add_argument('--cutmix',
                        default=False,
                        action='store_true',
                        help='Use cutmix data augmentation')
    parser.add_argument('--mixup',
                        default=False,
                        action='store_true',
                        help='Use mixup data augmentation')
    parser.add_argument('--cutmix-prob',
                        type=float,
                        default=1.0,
                        metavar='N',
                        help='CutMix probability (default is for ResNet. Use 0.5 for CIFAR10/100)')
    parser.add_argument('--alpha',
                        type=float,
                        default=.2,
                        metavar='N',
                        help='MixUp Alpha (default is for ResNet ImageNet. Use 1.0 for CIFAR10/100)')
    parser.add_argument('--beta',
                        type=float,
                        default=1.0,
                        metavar='N',
                        help='CutMix Beta (default is for ResNet ImageNet and WideResNet CIFAR10/100.)')
    parser.add_argument('--feature-extractor',
                        default="WideResNet28-10",
                        help='Backbone feature extractor (WideResNet28-10, WideResNet48-2)')
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='Train model')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='Evaluate a model model. args.resume required.')
    parser.add_argument('--fgsm',
                        default=False,
                        action='store_true',
                        help='Evaluate a model model under FGSM attack.')
    parser.add_argument('--jitter',
                        default=False,
                        action='store_true',
                        help='Evaluate a model model under color jitter.')
    parser.add_argument('--crop',
                        default=False,
                        action='store_true',
                        help='Evaluate a model model under random resized crop.')
    parser.add_argument('--corruption',
                        default=None,
                        choices=['brightness', 'contrast', 'frost', 'gaussian_noise',  'impulse_noise', 'motion_blur',  'pixelate',  'shot_noise',  'spatter', 'defocus_blur',  'elastic_transform',  'fog', 'gaussian_blur',  'glass_blur', 'jpeg_compression',  'saturate',  'snow', 'speckle_noise',  'zoom_blur'],
                        help='Evaluate a model model using this corruption mode.')
    parser.add_argument('--save',
                        default=False,
                        action='store_true',
                        help='Save checkpoint file every epoch')
    parser.add_argument('--imagenet-dir',
                        default="/data/imagenet",
                        help='Folder of imagenet dataset')
    parser.add_argument('--speech-commands-dir',
                        default='/data/speech/speech_commands/dataset',
                        help='Folder of imagenet dataset')
    parser.add_argument('--bg-noise-dir',
                        default='/data/speech/speech_commands/dataset/train/_background_noise_',
                        help='Folder of speech commnds dataset bg noise')
    parser.add_argument('--weights-dir',
                        default="weights",
                        help='Folder of model weights')
    parser.add_argument('--logs-dir',
                        default="logs",
                        help='Folder of debug logs')
    parser.add_argument('--checkpoints-dir',
                        default="checkpoints",
                        help='Checkpoint for restoring model for inference/resume training')
    parser.add_argument('--resume',
                        default=None,
                        help='Resume training using this weight file stored in checkpoint dir')
    parser.add_argument('--summary',
                        default=False,
                        action='store_true',
                        help='Print model summary')
    parser.add_argument('--dataset',
                        default="cifar10",
                        metavar='N',
                        help='Dataset for training classifier')
    parser.add_argument('--mimax',
                        default=False,
                        action='store_true',
                        help='Use MIMax')
    parser.add_argument('--seed', 
                        type=int, 
                        default=1, 
                        metavar='S',
                        help='Random seed')
    parser.add_argument('--results-dir',
                        default="results",
                        help='Folder of result files')
    args = parser.parse_args()
    return args
