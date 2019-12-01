'''
    Tensorboard logger
'''
import sys
import numpy as np

from tensorboardX import SummaryWriter

from . import utils

class Logger():
    def __init__(self, conf, cur_batch, num_batches):
        self.conf = conf
        self.summary = SummaryWriter()
        self.num_epochs = conf['num_epochs']
        self.num_batches = num_batches

    def log(self, losses=None, images=None):
        '''
            losses   : tensorboard에 visualize할 losses dictionary
            images   : tensorboard에 visualize할 images dictionary
        '''
        # Visualized logging for images
        for img_name, tensor in images.items(): # 현재 tensor는 H, W, C 형태 
            chw = tensor.transpose(1, 2).transpose(1, 3)
            self.summary.add_image(img_name, chw, self.conf['cur_iter'])
            
        # Visualize logging for losses
        for loss_name, loss in self.losses.items():
            self.summary.add_scalr(loss_name, loss, self.conf['cur_iter'])