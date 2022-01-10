import argparse
import torch

def get_args():
    # create args parser
    parser = argparse.ArgumentParser(description='Kws Trainer')

    # parameter for model
    parser.add_argument('--model', type=str, default='bcres')
    parser.add_argument('--metric', type=str, default='softmax')
    parser.add_argument('--loss', type=str, default='ce')

    # parameter for model's hyper parameters
    parser.add_argument('--n_keyword', type=int, default=12)
    parser.add_argument('--n_embed', type=int, default=512)
    parser.add_argument('--m', type=float, default=0.5)
    parser.add_argument('--s', type=float, default=64)

    # # parameter for training
    parser.add_argument('--no_shuffle', action='store_false')
    parser.add_argument('--no_evaluate', action='store_false')
    parser.add_argument('--clear_cache', action='store_true')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--log_iter', type=int, default=10)

    # parameter for optimizer
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='plateau')
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # parse args
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args
