import sys
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import BaseNetwork as Base
from util.Buffer import ImageBuffer
from util import utils

sys.path.append("..")

class InterpolationGAN(nn.Module):

    def __init__(self, params, is_train=True):
        # Device 설정
        if params.cuda:
            self.device = torch.device('cuda:{}'.format(params.gpu_id)) 
        else:
            self.device = torch.device('cpu')

        # Generator 생성 및 초기화
        self.G_S = Base.Generator(params.S_nc, params.T_nc)  # T를 S로 변환하는 Generator
        self.G_T = Base.Generator(params.T_nc, params.S_nc)  # S를 T로 변환하는 Generator
        if params.cuda:
            self.G_S.cuda()
            self.G_T.cuda()
        self.G_S.apply(utils.weights_init_normal)
        self.G_T.apply(utils.weights_init_normal)

        # Discriminator 생성 및 초기화
        if is_train:
            self.D_S = Base.Discriminator(params.S_nc)  # domain S를 구분하는 Discriminator
            self.D_T = Base.Discriminator(params.T_nc)  # domain T를 구분하는 Discriminator
            if params.cuda:
                self.D_S.cuda()
                self.D_T.cuda()
            self.D_S.apply(utils.weights_init_normal)
            self.D_T.apply(utils.weights_init_normal)

        # Model 구성요소 이름 저장
        if is_train:
            self.model_names = ['G_S', 'G_T', 'D_S', 'D_T']
            self.loss_names = ['G_S', 'D_S', 'Cycle_S', 'Ident_S', 'G_T', 'D_T', 'Cycle_T', 'Ident_T']
        else:
            self.model_names = ['G_S', 'G_T']
        # Visual name을 넣을 것인가 말 것인가.

        # Losses 및 Optimizer 생성
        if is_train:
            assert(params.S_nc == params.T_nc) # Identity Loss를 사용하려면 필요
            # 이전 생성 결과를 저장할 버퍼 --> 이거 왜 만드는 것이야
            self.save_fake_S = ImageBuffer(params.buf_size)
            self.save_fake_T = ImageBuffer(params.buf_size)
            # Losses
            self.criterion_GAN = nn.MSELoss()  # Adversarial loss at CycleGAN section 3.1
            self.criterion_cycle = nn.L1Loss()  # Cycle consistency loss at CycleGAN section 3.2
            self.criterion_identity = nn.L1Loss()  # Identity loss at CycleGAN section 5.2
            # Optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_S.parameters(), self.G_T.parameters()),
                                                lr=params.lr, betas=(params.beta, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_S.parameters(), self.D_T.parameters()),
                                                lr=params.lr, betas=(params.beta, 0.999))

            # LR schedulers
            '''lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                                  lr_lambda=LambdaLR(params.n_epochs, params.epoch,
                                                                                     params.decay_epoch).step)
            lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(paramsimizer_D_A,
                                                                    lr_lambda=LambdaLR(params.n_epochs, params.epoch,
                                                                                       params.decay_epoch).step)
            lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(paramsimizer_D_B,
                                                                    lr_lambda=LambdaLR(params.n_epochs, params.epoch,
                                                                                       params.decay_epoch).step)'''

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if params.cuda else torch.Tensor
        input_A = Tensor(params.batchSize, params.input_nc, params.size, params.size)
        input_B = Tensor(params.batchSize, params.output_nc, params.size, params.size)
        target_real = Variable(Tensor(params.batchSize).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(params.batchSize).fill_(0.0), requires_grad=False)

    def set_input(self, input):
        ''' 
            Iteration마다 DataLoader로부터 input을 받아서 unpack
        '''
        self.real_S = input['S'].to(self.device)
        self.real_T = input['T']


    def forward(self, input):
        pass

