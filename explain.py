import os
import time
import string
import argparse
import re

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def explain_all(model, opt):
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        explain_data(model, evaluation_loader, opt)
        print("Done explaining: ", eval_data)


def explain_data(model, evaluation_loader, opt):
    """ validation or evaluation """
    length_of_data = 0

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors[0].cpu().numpy()
        print(np.shape(image))
        exit(0)


        image = image_tensors.to(device)
        # For max length prediction

        start_time = time.time()
        preds = model(image, text=target, seqlen=converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, converter.batch_max_length)
        forward_time = time.time() - start_time
        cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
        preds_str = converter.decode(preds_index[:, 1:], length_for_pred)

        infer_time += forward_time

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]
            
            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            #if pred == gt:
            #    n_correct += 1



def explain(opt):
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length)
    # print(model)

    #if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./explain/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./explain/{opt.exp_name}/')

    """ evaluation """
    model.eval()
    with torch.no_grad():
        explain_all(model, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--imgH', type=int, default=224, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=224, help='the width of the input image')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ Model Architecture """
    choices = ["vit_small_patch16_224_str", "vit_base_patch16_224_str", "deit_tiny_patch16_224_str", "deit_small_patch16_224_str", "deit_base_patch16_224_str", "vit_base_patch16_384_str", "vit_base_patch32_384_str"]
    parser.add_argument('--TransformerModel', default=choices[0], help='Which vit/deit transformer model', choices=choices)

    parser.add_argument('--Transformer', action='store_true', help='Use end-to-end transformer')
    parser.add_argument('--Transformation', type=str, default=None, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default=None, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default=None, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default=None, help='Prediction stage. CTC|Attn')
    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

    explain(opt)
