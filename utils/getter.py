import torch

from models.metrics import *
from models.losses import *
from models.model import *

def get_embedding_model(args):
    models = {
        # 'tdnn': Xvector(args).to(args.device),
        # 'attn': Xattention(args).to(args.device),
        'bcres': BCResNet(args).to(args.device)
    }
    try:
        return models[args.model]
    except:
        raise NotImplementedError

def get_model(args):
    models = {
        'softmax': SoftMax(get_embedding_model(args), args).to(args.device),
        'adacos': AdaCos(get_embedding_model(args), args).to(args.device),
        'arcface': ArcFace(get_embedding_model(args), args).to(args.device),
        'cosface': CosFace(get_embedding_model(args), args).to(args.device),
        'sphereface': SphereFace(get_embedding_model(args), args).to(args.device),
    }
    try:
        return models[args.metric]
    except:
        raise NotImplementedError

def get_criterion(args):
    loss = {
        'ce': torch.nn.CrossEntropyLoss(),
        'ces': LabelSmoothingCrossEntropy(smoothing=0.1),
        'focal': FocalLoss(gamma=2)
    }
    try:
        return loss[args.loss]
    except:
        raise NotImplementedError

def get_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=0.000002)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=0.000002)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=0.001,
                                    nesterov=True)
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(optimizer, args):
    schedulers = {
        'none': None,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max'),
    }
    return schedulers[args.scheduler]

def print_summary(model):
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Non-trainable parameters:', sum(p.numel() for p in model.parameters() if not p.requires_grad))


