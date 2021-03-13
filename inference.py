import os
import argparse
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import string

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer_all(model, converter, opt):
    #eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
    #                  'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    eval_data_list = ['IC13_1015', 'CUTE80']

    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        dataset = eval_data
        os.makedirs(f'./{opt.dir}/{eval_data}', exist_ok=True)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        infer_data(model, evaluation_loader, converter, dataset, opt)
        print("Done infering: ", dataset)


def infer_data(model, evaluation_loader, converter, dataset, opt):
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        image = image_tensors.to(device)
        image_hash = (hex(hash(image.cpu().numpy().tobytes()))[2:8]).upper()

        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * 1)

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        
        else:
            preds = model(image, text_for_pred, is_train=False)

            preds = preds[:, :text_for_loss.shape[1] - 1, :]

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        #target = converter.encode(labels)
        #preds = model(image, text_for_pred)
        #_, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        #preds_index = preds_index.view(-1, opt.batch_max_length + 1)
        #preds_str = converter.decode(preds_index[:, 0:], length_for_pred)
        if 'Attn' in opt.Prediction:
            pred_EOS = preds_str[0].find('[s]')
            pred = preds_str[0][:pred_EOS]  # prune after "end of sentence" token ([s])
        else:
            pred = preds_str[0]
        print(f'Pred: {pred}')
        print(f'GT:   {labels[0]}')

        img = image_tensors[0].squeeze()
        img = img.cpu().numpy()
        img = (((img + 1) * 0.5) * 255).astype(np.uint8)
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
        mask_name = "{}/{}/{}-pred-{}-{}.png".format(opt.dir,
                                                     dataset,      
                                                     labels[0], 
                                                     pred,
                                                     image_hash)

        cv2.imwrite(mask_name, img)


def infer(opt):
    if opt.Transformer:
        converter = TokenLabelConverter(opt)
    elif 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    """ keep evaluation model and result logs """
    os.makedirs(f'./{opt.dir}/{opt.exp_name}', exist_ok=True)

    """ evaluation """
    model.eval()
    infer_all(model, converter, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--calculate_infer_time', action='store_true', help='calculate inference timing')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformer', action='store_true', help='Use end-to-end transformer')
    choices = ["vit_small_patch16_224_str", "vit_base_patch16_224_str", "deit_tiny_patch16_224_str", "deit_tiny_patch16_224_str_gray", "deit_small_patch16_224_str", "deit_base_patch16_224_str", "vit_base_patch16_384_str", "vit_base_patch32_384_str"]
    parser.add_argument('--TransformerModel', default=choices[0], help='Which vit/deit transformer model', choices=choices)
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--flops', action='store_true', help='calculates approx flops (may not work)')

    parser.add_argument('--issel_aug', action='store_true', help='Select augs')
    parser.add_argument('--sel_prob', type=float, default=1.0, help='Probability of applying augmentation')
    parser.add_argument('--pattern', action='store_true', help='Pattern group')
    parser.add_argument('--warp', action='store_true', help='Warp group')
    parser.add_argument('--geometry', action='store_true', help='Geometry group')
    parser.add_argument('--weather', action='store_true', help='Weather group')
    parser.add_argument('--noise', action='store_true', help='Noise group')
    parser.add_argument('--blur', action='store_true', help='Blur group')
    parser.add_argument('--camera', action='store_true', help='Camera group')
    parser.add_argument('--process', action='store_true', help='Image processing routines')

    parser.add_argument('--scheduler', action='store_true', help='Use lr scheduler')

    parser.add_argument('--intact_prob', type=float, default=0.25, help='Probability of not applying augmentation')
    parser.add_argument('--augs_num', type=int, default=7, help='Number of data augment groups to apply')
    parser.add_argument('--isrand_aug', action='store_true', help='Use RandAug')
    parser.add_argument('--isprio_rand_aug', action='store_true', help='Use prioritized RandAug')

    parser.add_argument('--fast_acc', action='store_true', help='Fast average accuracy computation')

    parser.add_argument('--dir', type=str, default="sample_predictions", help='Results folder')
    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

    opt.eval = True
    infer(opt)
