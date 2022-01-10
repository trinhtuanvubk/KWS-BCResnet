import argparse
import torch

def get_args():
    # create args parser
    parser = argparse.ArgumentParser(description='Kws Trainer')

    # parameter for dataset
    # parser.add_argument('--data_dir', type=str, default='data')
    # parser.add_argument('--no_extract', action='store_false')

    # parameter for model
    parser.add_argument('--model', type=str, default='bcres')
    parser.add_argument('--metric', type=str, default='softmax')
    parser.add_argument('--loss', type=str, default='ce')

    # parameter for model's hyper parameters
    parser.add_argument('--n_mels', type=int, default=40)
    parser.add_argument('--n_fft', type=int, default=400)
    # parser.add_argument('--cnn_channel', type=str, default='512,512,512,512,1500')
    # parser.add_argument('--cnn_kernel', type=str, default='5,3,3,1,1')
    # parser.add_argument('--cnn_dilation', type=str, default='1,2,3,1,1')
    parser.add_argument('--n_embed', type=int, default=512)
    # parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_keyword', type=int, default=12)

    # parameter for loss's hyper parameters
    parser.add_argument('--m', type=float, default=0.5)
    parser.add_argument('--s', type=float, default=64)


    # # parameter for training
    parser.add_argument('--no_shuffle', action='store_false')
    parser.add_argument('--no_evaluate', action='store_false')
    parser.add_argument('--clear_cache', action='store_true')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--log_iter', type=int, default=10)
    # parser.add_argument('--num_worker', type=int, default=2)
    # parser.add_argument('--no_pin_memory', action='store_false')
    # parser.add_argument('--clip_grad_norm', type=float, default=5.0)
    # parser.add_argument('--limit_train_batch', type=int, default=-1)
    # parser.add_argument('--limit_val_batch', type=int, default=-1)

    # parameter for optimizer
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='plateau')
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # parameter for visualize
    # parser.add_argument('--plot', action='store_true')
    # parser.add_argument('--top_k', type=int, default=5)

    # parse args
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args
