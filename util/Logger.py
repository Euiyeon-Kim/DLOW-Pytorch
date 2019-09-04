'''
    Have to open visdom server first
    'python -m visdom.server' or 'visdom'
'''

import sys
import time
import numpy as np

from visdom import Visdom

from . import utils

sys.path.append("..")

class Logger():
    def __init__(self, num_epochs, num_batches):
        self.viz = Visdom()
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.cur_epoch = 1
        self.cur_batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.terms = {}                                # Running average로 매 배치마다 terminal에 logging
        self.losses = {}                               # Running average로 매 epoch마다 visualize할 data
        self.visualize_losses = {}                     # Windows for visualized losses
        self.visualize_images = {}                     # Windows for visualized images

    def log(self, terms=None, losses=None, images=None):
        '''
            terms:  terminal에 logging할 정보 dictionary
            losses: visdom에 visualize할 losses dictionary
            images: visdom에 visualize할 images dictionary
        '''
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\Epoch %03d/%03d [%04d/%04d] -- '\
                         %(self.cur_epoch, self.num_epochs, self.cur_batch, self.num_batches))

        # Logging for terminal - for evey batch
        for idx, log_name in enumerate(terms.keys()):      # Index가 필요하므로 enumerate
            if log_name not in self.terms:
                self.terms[log_name] = terms[log_name]
            else:
                self.terms[log_name] += terms[log_name]
            if (idx+1) == len(terms):
                sys.stdout.write('%s: %0.4f -- '%(log_name, self.terms[log_name]/self.cur_batch))
            else:
                sys.stdout.write('%s: %0.4f | '%(log_name, self.terms[log_name]/self.cur_batch))

        # Running average for visualized loss logging
        for loss_name, loss in losses.items():
            if loss_name not in self.losses:
                self.losses[loss_name] = loss
            else:
                self.losses[loss_name] += loss

        # Visualized logging for images - every batch
        for img_name, tensor in images.items():
            if img_name not in self.visualize_images:       # 새로운 이미지 일 경우 윈도우 생성
                self.visualize_images[img_name] = self.viz.image(utils.tensor2img(tensor.data), opts={'title':img_name})
            else:                                           # 이전 윈도우의 이미지 교체
                self.viz.image(utils.tensor2img(tensor.data), win=self.visualize_images[img_name], opts={'title':img_name})

        # Visualize logging for losses - every epoch
        if self.cur_batch % self.num_batches == 0:
            for loss_name, loss in self.losses.items():
                if loss_name not in self.visualize_losses:  # 새로운 loss일 경우 윈도우 생성
                    self.visualize_losses[loss_name] = self.viz.line(X=np.array([self.cur_epoch]), Y=np.array([loss/self.num_batches]),
                                                                     opts={'title':loss_name, 'xlabel':'epochs', 'ylabel':loss_name})
                else:                                       # 이전 윈도우에 info append
                    self.viz.line(X=np.array([self.cur_epoch]), Y=np.array([loss/self.num_batches]), win=self.visualize_losses[loss_name], update='append')
                self.losses[loss_name] = 0.0

            self.cur_epoch += 1
            self.cur_batch = 1
            sys.stdout.write('\n')
        else:
            self.cur_batch += 1
