import os

def get_ckpt_folder(args,config):
    folder = f'ckpt/{args.model}_{args.metric}_{args.loss}_{config.n_keyword}/checkpoints/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def get_checkpoint_path(args,config):
    path = os.path.join(get_ckpt_folder(args,config), 'model.ckpt')
    return path

def get_log_folder(args,config):
    folder = get_ckpt_folder(args,config)
    folder = folder.replace('checkpoints', 'logs')
    return folder

def get_visualize_folder(args):
    root_path = f'visualize/{args.model}_{args.metric}_{args.loss}_{args.n_keyword}/'
    sub_folder = ['train', 'test']
    for folder in sub_folder:
        path = os.path.join(root_path, folder)
        if not os.path.exists(path):
            os.makedirs(path)
    return root_path