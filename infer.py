import torch
import string
import numpy as np
from infer_utils import TokenLabelConverter, NormalizePAD, get_args
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer(args):
    converter = TokenLabelConverter(args)
    args.num_class = len(converter.character)
    transform = NormalizePAD((args.input_channel, args.imgH, args.imgW))
    img = Image.open(args.image).convert('L')
    img = img.resize((args.imgW, args.imgH), Image.BICUBIC)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    
    model = torch.load(args.model)
    model.eval()
    with torch.no_grad():
        pred = model(img, seqlen=converter.batch_max_length)
        _, pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
        pred_index = pred_index.view(-1, converter.batch_max_length)
        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] ) #.to(device)
        pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
        pred_EOS = pred_str[0].find('[s]')
        pred_str = pred_str[0][:pred_EOS]

    return pred_str


if __name__ == '__main__':
    args = get_args()
    args.character = string.printable[:-6] 
    print(infer(args))
