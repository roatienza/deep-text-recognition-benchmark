import os
import argparse
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from utils import TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from model import Model

from modules.vit_rollout import VITAttentionRollout
from modules.vit_grad_rollout import VITAttentionGradRollout

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def explain_all(model, converter, opt):
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    eval_data_list = ['SVT'] # 'IC03_860', 'IC03_867', 'IC13_857',

    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        dataset = eval_data
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        explain_data(model, evaluation_loader, converter, dataset, opt)
        print("Done explaining: ", eval_data)


def explain_data(model, evaluation_loader, converter, dataset, opt):
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        image = image_tensors.to(device)

        target = converter.encode(labels)
        preds = model(image, text=target, seqlen=converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, converter.batch_max_length)
        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] ).to(device)
        preds_str = converter.decode(preds_index[:, 1:], length_for_pred)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        if opt.grad:
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=opt.discard_ratio)
            mask = grad_rollout(image, opt.token)
            mask_name = "{}/{}-mask-grad_rollout_cat{}_{:.3f}_{}.png".format(opt.dir, labels[0], opt.token, opt.discard_ratio, opt.head_fusion)
            # image_name = "{}/{}-grad_rollout_cat{}_{:.3f}_{}.png".format(opt.dir, labels[0], opt.token, opt.discard_ratio, opt.head_fusion)
        else:
            attention_rollout = VITAttentionRollout(model, 
                                                    head_fusion=opt.head_fusion, 
                                                    discard_ratio=opt.discard_ratio, 
                                                    token=opt.token,
                                                    seqlen=converter.batch_max_length)
            mask = attention_rollout(image)
            mask_name = "{}/{}/{}-rollout-token{}-{:.2f}_{}.png".format(opt.dir,
                                                                        dataset,      
                                                                        labels[0], 
                                                                        opt.token,
                                                                        opt.discard_ratio, 
                                                                        opt.head_fusion)
            os.makedirs(f'./{opt.dir}/{dataset}', exist_ok=True)
            #image_name = "{}/{}-attention_rollout_{:.3f}_{}.png".format(opt.dir, labels[0], opt.discard_ratio, opt.head_fusion)

        image = image_tensors[0].squeeze()
        image = image.cpu().numpy()
        image = (((image + 1) * 0.5) * 255).astype(np.uint8)
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = show_mask_on_image(image, mask)
        cv2.imshow("Mask Image", mask)
        cv2.imwrite(mask_name, mask)
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]
            print(f'Pred: {pred}')
            print(f'GT:   {gt}')

        cv2.waitKey(-1)
        if i==10:
            exit(0)



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
    #for name, module in model.named_modules():
    #    print(name)
    #exit(0)
    """ keep evaluation model and result logs """
    os.makedirs(f'./{opt.dir}/{opt.exp_name}', exist_ok=True)
    #os.system(f'cp {opt.saved_model} ./{opt.dir}/{opt.exp_name}/')

    """ evaluation """
    model.eval()
    explain_all(model, converter, opt)


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
    parser.add_argument('--dir', type=str, default="interpret", help='Results folder of interpretability')
    parser.add_argument('--head_fusion', type=str, default='mean',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--token', type=int, default=None,
                        help='The token for rollout')
    parser.add_argument('--grad', action='store_true', help='grad rollout')
    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

    explain(opt)
