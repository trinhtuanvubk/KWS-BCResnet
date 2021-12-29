
from torch import nn
import torch
from train.args import get_args
import utils
def test(args):
    model = utils.get_embedding_model(args)
    model.print_summary(model)
    x = torch.randn(3, 101, 40)
    x = model(x)
    print(x.shape)

    model = model.get_model(args)
    model.print_summary(model)

if __name__=='__main__':
    args = get_args()
    test(args=args)
    

    