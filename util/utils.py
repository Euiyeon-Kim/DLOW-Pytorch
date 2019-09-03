import os
import json
import logging
import shutil
import numpy as np

import torch
import torch.nn as nn
# =======================================


from torch.optim import lr_scheduler
from torch.autograd import Variable

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Ignore logging messages which is less severe than level

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to a console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path): # d=dictionary
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint_path):
    file_path = os.path.join(checkpoint_path, 'last.pth.tar')

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    torch.save(state, file_path) # Save checkpoint

    if is_best: # If results was best, save checkpoint at best.pth.tar
        shutil.copyfile(file_path, os.path.join(checkpoint_path, 'best.pth.tar'))


def load_checkpoint(checkpoint_path, S2T, T2S, D_S, D_T, G_paramsimizer=None, D_paramsimizer=None):
    if not os.path.exists(checkpoint_path):
        raise ("Checkpoint doesn't exist at {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)
    # state_dict : model의 learnable parameter 상태를 dictionary로 표현
    # layer name : parameter tensor의 형태
    S2T.load_state_dict(checkpoint['S2T_state_dict'])
    T2S.load_state_dict(checkpoint['T2S_state_dict'])
    D_S.load_state_dict(checkpoint['D_S_state_dict'])
    D_T.load_state_dict(checkpoint['D_T_state_dict'])

    if G_paramsimizer:
        G_paramsimizer.load_state_dict(checkpoint['G_paramsimizer_state_dict'])
    if D_paramsimizer:
        D_paramsimizer.load_state_dict(checkpoint['D_paramsimizer_state_dict'])

    return checkpoint


def init_weights(net, init_type='normal', init_gain=0.02):

    def actual_init(model):
        classname = model.__class__.__name__
        if hasattr(model, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(model.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(model.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(model.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(model.weight.data, 1.0, init_gain)
            nn.init.constant_(model.bias.data, 0.0)

    net.apply(actual_init)


# Referenced https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
def get_scheduler(paramsimizer, args):
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.epoch_count - args.start_decay) / float(args.decay_cycle + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(paramsimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(paramsimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(paramsimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(paramsimizer, T_max=args.start_decay, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def tensor2img(tensor):
    '''
        Input이 tensor라면 numpy image array로 변환
        Dataloader에서 data를 읽어올 때 pixel값의 범위를 [-1, 1]로 normalize했으므로 이를 재변환
    '''
    img = ((tensor[0].cpu().float().numpy()) + 1) / 2.0 * 255.0
    if img.shape[0]==1:
        img = np.tile(img, (3, 1, 1))
    return img.astype(np.uint8)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

