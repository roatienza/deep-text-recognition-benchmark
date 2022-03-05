'''
Script to use ViTSTR to convert scene text image to text.

Usage:
    python3 infer.py --image demo_image/demo_1.png --model https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug_infer.pth

--image: path to image file to convert to text

To convert DataParallel model to cpu model:
https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/12
Basically: 
   1) Load the model in DataParallel
   2) Get a new state dict
   3) Create a new model that is not DataParallel using state dict fr 2)
   4) Load the new state dict
   5) Save the model
For generating new state dict
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
'''


import torch
import string
import validators
from infer_utils import TokenLabelConverter, NormalizePAD,  ViTSTRFeatureExtractor
from infer_utils import get_args


def infer(args):
    converter = TokenLabelConverter(args)
    args.num_class = len(converter.character)
    img = ViTSTRFeatureExtractor()(args.image)
    
    if validators.url(args.model):
        checkpoint = args.model.rsplit('/', 1)[-1]
        torch.hub.download_url_to_file(args.model, checkpoint)
    else:
        checkpoint = args.model
    model = torch.load(checkpoint)
    model.eval()
    with torch.no_grad():
        pred = model(img, seqlen=converter.batch_max_length)
        _, pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
        pred_index = pred_index.view(-1, converter.batch_max_length)
        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] )
        pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
        pred_EOS = pred_str[0].find('[s]')
        pred_str = pred_str[0][:pred_EOS]

    return pred_str


if __name__ == '__main__':
    args = get_args()
    args.character = string.printable[:-6] 
    print(infer(args))
