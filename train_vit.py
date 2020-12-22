'''
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
from timm.models.vision_transformer import VisionTransformer
from transform import data_augment
from utilities.misc import get_device, AverageMeter
from utilities.metrics import accuracy
from utilities.ui import progress_bar

import datetime
import os
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.nn as nn
import torchvision.utils

#from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, resume_checkpoint, load_checkpoint, convert_splitbn_model
#from timm.utils import *
#from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
#from timm.optim import create_optimizer
#from timm.scheduler import create_scheduler
#from timm.utils import ApexScaler, NativeScaler


class Loader:
    def __init__(self,
                 root='./data',
                 batch_size=512, 
                 dataset=datasets.CIFAR10, 
                 transform={'train':transforms.ToTensor(), 'test':transforms.ToTensor()},
                 device=None,
                 dataset_name="cifar10",
                 shuffle_test=False,
                 corruption=None):
        super(Loader, self).__init__()
        self.test = None
        self.train = None
        self._build(root,
                    batch_size, 
                    dataset, 
                    transform, 
                    device, 
                    dataset_name,
                    shuffle_test,
                    corruption)

    
    def _build(self,
               root,
               batch_size, 
               dataset, 
               transform, 
               device,
               dataset_name,
               shuffle_test,
               corruption):
        DataLoader = torch.utils.data.DataLoader
        workers = torch.cuda.device_count() * 4
        if "cuda" in str(device):
            print("num_workers: ", workers)
            kwargs = {'num_workers': workers, 'pin_memory': True}
        else:
            kwargs = {}

        if dataset_name == "svhn" or dataset_name == "svhn-core":
            x_train = dataset(root=root,
                              split='train',
                              download=True,
                              transform=transform['train'])

            if dataset_name == "svhn":
                x_extra = dataset(root=root,
                                  split='extra',
                                  download=True, 
                                  transform=transform['train'])
                x_train = ConcatDataset([x_train, x_extra])

            x_test = dataset(root=root,
                             split='test',
                             download=True,
                             transform=transform['test'])
        elif dataset_name == "imagenet":
            x_train = dataset(root=root,
                              split='train', 
                              transform=transform['train'])
            if corruption is None:
                x_test = dataset(root=root,
                                 split='val', 
                                 transform=transform['test'])
            else:
                root = os.path.join(root, corruption)
                corrupt_test = []
                for i in range(1, 6):
                    folder = os.path.join(root, str(i))
                    x_test = datasets.ImageFolder(root=folder,
                                                  transform=transform['test'])
                    corrupt_test.append(x_test)
                x_test = ConcatDataset(corrupt_test)

        elif dataset_name == "speech_commands":
            x_train = dataset(root=root,
                              split='train', 
                              transform=transform['train'])
            x_val = dataset(root=root,
                            split='valid', 
                            transform=transform['test'])
            x_test = dataset(root=root,
                             split='test', 
                             transform=transform['test'])

            self.val = DataLoader(x_val,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  **kwargs)
        else:
            x_train = dataset(root=root,
                              train=True,
                              download=True,
                              transform=transform['train'])

            x_test = dataset(root=root,
                             train=False,
                             download=True,
                             transform=transform['test'])

        self.train = DataLoader(x_train,
                                shuffle=True,
                                batch_size=batch_size,
                                **kwargs)

        self.test = DataLoader(x_test,
                               shuffle=shuffle_test,
                               batch_size=batch_size,
                               **kwargs)

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text



class Classifier:
    def __init__(self,
                 args, 
                 model, 
                 dataloader,
                 device=get_device()):
        super(Classifier, self).__init__()
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.best_top1 = 0
        self.best_top5 = 0
        self.best_epoch = 0 
        self.milestones = [30, 60, 80]
        self._build()


    def get_model_name(self):
        model_name = self.args.dataset + "-baseline-"
        model_name += self.args.model + "-"
        return model_name


    def _log_loss(self, epoch, ce, entropy, dl):
        folder = self.args.logs_dir
        os.makedirs(folder, exist_ok=True)
        model_name = self.get_model_name()
        filename = model_name + "train-loss.log"
        path = os.path.join(folder, filename)
        filename = open(path, "a+")
        if epoch == 1:
            logs = ["Epoch,CE,Entropy,L1"]
            logs.append("%d,%f,%f,%f" % (epoch, ce, entropy, dl))
        else:
            logs = ["%d,%f,%f,%f" % (epoch, ce, entropy, dl)]

        for log in logs:
            filename.write(log)
            filename.write("\n")
        filename.close()


    def _log_acc(self, epoch, top1, top5, is_val=False, val_name=None):
        folder = self.args.logs_dir
        os.makedirs(folder, exist_ok=True)
        model_name = self.get_model_name()
        if is_val:
            filename = model_name + "val-acc.log"
        elif val_name is not None:
            filename = model_name + val_name + "-acc.log"
        else:
            filename = model_name + "test-acc.log"
        path = os.path.join(folder, filename)
        filename = open(path, "a+")
        if epoch == 1:
            logs = ["---------%s--------%s---------" % \
                    (model_name, datetime.datetime.now())]
            logs.append("Epoch,Top1,Top5")
            logs.append("%d,%f,%f" % (epoch, top1, top5))
        else:
            logs = ["%d,%f,%f" % (epoch, top1, top5)]

        for log in logs:
            filename.write(log)
            filename.write("\n")
        filename.close()

    def _log(self, top1=None, top5=None, verbose=True):

        folder = self.args.logs_dir
        os.makedirs(folder, exist_ok=True)
        model_name = self.get_model_name()
        if top1 is None:
            filename = model_name + "start.log"
        else:
            filename = model_name + "end.log"
        path = os.path.join(folder, filename)
        filename = open(path, "a+")
        logs = ["---------%s--------%s---------" % \
                (model_name, datetime.datetime.now())]
        logs.append("Device: %s" % self.device)
        logs.append("Dataset: %s" % self.args.dataset)
        logs.append("Number of classes: %d" % self.args.num_classes)
        logs.append("Backbone: %s" % self.args.model)
        logs.append("Batch size: %d" % self.args.batch_size)
        logs.append("Weight decay: %f" % self.args.weight_decay)
        logs.append("LR: %f" % self.args.lr)
        logs.append("Epochs: %d" % self.args.epochs)
        #logs.append("Dropout: %f" % self.args.dropout)
        logs.append("---------%s--------%s---------" % \
                    (model_name, datetime.datetime.now()))

        for log in logs:
            filename.write(log)
            filename.write("\n")
            if verbose:
                print(log)
        filename.close()


    def assign_lr_scheduler(self, last_epoch=-1):
        #if self.args.cosine:
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, last_epoch=last_epoch)
        #else:
        #        if args.decay_type == "cosine":
        #self.scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    def _build(self, init_weights=False):
        self.model = self.model.to(self.device)

        if "cuda" in str(self.device):
            self.model = torch.nn.DataParallel(self.model)
            print("Data parallel:", self.device)

        cudnn.benchmark = True

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.args.lr,
                                   momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay,
                                   nesterov=True)
        self.assign_lr_scheduler()
        self._log()

        #lr = 0.5 * initial_lr * (
        #        1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))

    def prepare_train(self, best_top1, best_top5, best_model, epoch):
        best_model = "None" if best_model is None else best_model
        info = "\nEpoch %d(%d), PID %d, "
        info += "Dataset: %s, Best Top 1: %0.2f%%, Best Top 5: %0.2f%% Best Model: %s"
        print(info % (epoch, self.args.epochs, os.getpid(), self.args.dataset, best_top1, best_top5, best_model))
        self.model.train()


    def train(self, best_top1, best_top5, best_model, epoch):
        self.prepare_train(best_top1, best_top5, best_model, epoch)
        lr = self.scheduler.get_last_lr()
        correct = 0
        total = 0
        losses = AverageMeter()

        ce_loss = nn.CrossEntropyLoss().to(self.device)       
        for i, data in enumerate(self.dataloader.train):
            image, target = data
            x = image.to(self.device)
            target = target.to(self.device)

            y = self.model(x)
            self.optimizer.zero_grad()
            loss = ce_loss(y, target)
            loss.backward()
            self.optimizer.step()

            losses.update(loss.float().mean().item())
                
            _, predicted = y.max(1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            acc = correct * 100. / total

            progress_bar(i,
                         len(self.dataloader.train), 
                         'CE: %.4f | Top 1 Acc: %0.2f%% | LR: %.2e'
                         % (losses.avg, acc, lr[0]))
        
        return losses.avg
        

    def eval(self, epoch=0, is_val=False, val_name=None):
        self.model.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        extra = ""
        if is_val:
            loader = self.dataloader.val
            dset = "val"
        else:
            loader = self.dataloader.test
            dset = "test"
        with torch.no_grad():
            for i, data in enumerate(loader):
                x, target = data
                x = x.to(self.device)
                target = target.to(self.device)

                y = self.model(x)
                acc1, acc5 = accuracy(y, target, (1, 5))
                top1.update(acc1[0], x.size(0))
                top5.update(acc5[0], x.size(0))

                progress_bar(i,
                             len(self.dataloader.test), 
                             '%s%s %s %s accuracy: Top 1: %0.2f%%, Top 5: %0.2f%%'
                             % (self.args.model, extra, self.args.dataset, dset, top1.avg, top5.avg))
                
            if self.best_top1 > 0 and not is_val:
                info = "Epoch %d top 1 accuracy: %0.2f%%"
                info += ", Old best top 1 accuracy: %0.2f%% at epoch %d"
                data = (epoch, top1.avg, self.best_top1, self.best_epoch)
                print(info % data)

            if top1.avg > self.best_top1 and not is_val:
                self.best_top1 = top1.avg.float().item()
                self.best_top5 = top5.avg.float().item()
                self.best_epoch = epoch
                info = "New best top1: %0.2f%%, top5: %0.2f%%"
                print(info % (self.best_top1, self.best_top5))
                folder = self.args.weights_dir
                os.makedirs(folder, exist_ok=True)
                self.best_model = self.get_model_name()
                self.best_model += str(round(self.best_top1,2)) 
                self.best_model += ".pth"
                path = os.path.join(folder, self.best_model)
                self.save_checkpoint(epoch, path=path, is_best=True)

            if self.args.save:
                self.save_checkpoint(epoch)
            
            self._log_acc(epoch, top1.avg.float().item(), top5.avg.float().item(), is_val=is_val, val_name=val_name)

        return self.best_top1, self.best_top5, self.best_model


    def save_checkpoint(self, epoch, path=None, is_best=False):
        if not is_best:
            folder = self.args.checkpoints_dir
            os.makedirs(folder, exist_ok=True)
            filename = self.get_model_name() + "epoch-" + str(epoch) + ".pth"
            path = os.path.join(folder, filename)

        print("Saving checkpoint ... ", path)
        checkpoint = {'epoch': epoch,
                      'best_top1': self.best_top1,
                      'best_top5': self.best_top5,
                      'best_epoch': self.best_epoch,
                      'best_model': self.best_model,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict' : self.optimizer.state_dict(),
                      'scheduler_state_dict' : self.scheduler.state_dict(),
                     }
        torch.save(checkpoint, path)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    #opt = parser.parse_args()


    has_apex = False
    has_native_amp = False
    torch.backends.cudnn.benchmark = True
    _logger = logging.getLogger('train')

    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')


    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Dataset / Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train (default: "countception"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                        help='ratio of validation batch size to training batch size (default: 1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')



    # Learning rate schedule parameters
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='const',
                        help='Random erase mode (default: "const")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-tf', action='store_true', default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='save images of input bathes every log interval for debugging')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')

    parser.add_argument('--resume', action='store_true', default=False, help="resume")
    parser.add_argument('--dataset',
                        default="cifar10",
                        metavar='N',
                        help='Dataset for training classifier')
    parser.add_argument('--weights-dir',
                        default="weights",
                        help='Folder of model weights')
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='Train model')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='Evaluate a model model. args.resume required.')
    parser.add_argument('--imagenet-dir',
                        default="/data/imagenet",
                        help='Folder of imagenet dataset')
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
    parser.add_argument('--logs-dir',
                        default="logs",
                        help='Folder of debug logs')
    parser.add_argument('--save',
                        default=False,
                        action='store_true',
                        help='Save checkpoint file every epoch')

    args = parser.parse_args()


    folder = args.weights_dir
    os.makedirs(folder, exist_ok=True)
    root = './data'
    transform=[transforms.ToTensor(), transforms.ToTensor()],
    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10
        args.num_classes = 10
    elif args.dataset == "cifar100":  
        dataset = datasets.CIFAR100
        args.num_classes = 100
    elif args.dataset == "svhn" or args.dataset == "svhn-core":  
        dataset = datasets.SVHN
        args.num_classes = 10
    elif args.dataset == "imagenet":
        dataset = datasets.ImageNet
        root = args.imagenet_dir
        args.num_classes = 1000
        # fr CutMix https://arxiv.org/pdf/1905.04899.pdf
        length = 112
    else:
        ValueError("Not supported dataset")

    transform = data_augment(dataset=args.dataset)
    dataloader = Loader(transform=transform,
                        device=get_device(),
                        batch_size=args.batch_size,
                        dataset=dataset)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)

    #print(model)


    classifier = Classifier(args, model=model, dataloader=dataloader, device=get_device())

    start_epoch = 1
    end_epoch = args.epochs + 1
    if args.resume:
        folder = args.checkpoints_dir
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, args.resume)
        print("Resuming from checkpoint '%s'" % path)
        checkpoint = torch.load(path)
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.model.to(get_device())
        classifier.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        classifier.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['epoch']
        classifier.assign_lr_scheduler(last_epoch=last_epoch-1)
        start_epoch = last_epoch + 1
        args.best_top1 = checkpoint['best_top1']
        args.best_top5 = checkpoint['best_top5']
        args.best_model = checkpoint['best_model']

        classifier.best_top1 = args.best_top1
        classifier.best_top5 = args.best_top5
        classifier.best_model = args.best_model
        classifier.best_epoch = checkpoint['best_epoch']

        if args.eval:
            val_name = None
            if args.corruption is not None:
                val_name = args.corruption
                print("Corruption mode:", args.corruption)
            classifier.eval(start_epoch - 1, val_name=val_name)

    if args.train:
        best_top1 = args.best_top1
        best_top5 = args.best_top5
        best_model = args.best_model
        for epoch in range(start_epoch, end_epoch):
            start_time = datetime.datetime.now()
            loss = classifier.train(best_top1, best_top5, best_model, epoch)
            top1, top5, model = classifier.eval(epoch)
            classifier.scheduler.step()
            if top1 > best_top1:
                best_top1 = top1
                best_top5 = top5
                best_model = model
            elapsed_time = datetime.datetime.now() - start_time
            print("Elapsed time: %s" % elapsed_time)
        classifier._log(top1=top1, top5=top5, verbose=True)

