import os
import sys
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import BaseNetwork as Base
from util.Buffer import ImageBuffer
from util import utils

sys.path.append("..")

class CycleGAN(nn.Module):

    def __init__(self, params, is_train=True):
        super(CycleGAN, self).__init__()
        self.params = params
        # Device 설정
        if params.cuda: # 후에 multi-GPU coding할 수 있으면 적용
            self.device = torch.device('cuda:{}'.format(params.gpu_id)) 
        else:
            self.device = torch.device('cpu')

        # Generator 생성 및 초기화
        self.G_S = Base.Generator(params.T_nc, params.S_nc, params.ngf, params.use_dropout, params.n_res_blocks)  # T를 S로 변환하는 Generator
        self.G_T = Base.Generator(params.S_nc, params.T_nc, params.ngf, params.use_dropout, params.n_res_blocks)  # S를 T로 변환하는 Generator
        if params.cuda:
            self.G_S.cuda()
            self.G_T.cuda()
        utils.init_weights(self.G_S)
        utils.init_weights(self.G_T)

        # Discriminator 생성 및 초기화
        if is_train:
            self.D_S = Base.Discriminator(params.S_nc, params.ndf)  # domain S를 구분하는 Discriminator
            self.D_T = Base.Discriminator(params.T_nc, params.ndf)  # domain T를 구분하는 Discriminator
            if params.cuda:
                self.D_S.cuda()
                self.D_T.cuda()
            utils.init_weights(self.D_S)
            utils.init_weights(self.D_T)
            Tensor = torch.cuda.FloatTensor if params.cuda else torch.Tensor
            self.real = Variable(Tensor(params.batch_size).fill_(1.0), requires_grad=False)
            self.fake = Variable(Tensor(params.batch_size).fill_(0.0), requires_grad=False)

        # Model 구성요소 이름 저장
        if is_train:
            self.model_names = ['G_S', 'G_T', 'D_S', 'D_T']
            self.loss_names = ['G_S', 'D_S', 'Cycle_S', 'Ident_S', 'G_T', 'D_T', 'Cycle_T', 'Ident_T']
        else:
            self.model_names = ['G_S', 'G_T']

        # Losses 및 Optimizer 생성
        if is_train:
            assert(params.S_nc == params.T_nc)          # Identity Loss를 사용하려면 필요
            # Buffers to save previously generated images
            self.save_fake_S = ImageBuffer(params.buf_size)
            self.save_fake_T = ImageBuffer(params.buf_size)
            # Losses
            self.criterion_GAN = nn.MSELoss()           # Adversarial loss at CycleGAN section 3.1
            self.criterion_cycle = nn.L1Loss()          # Cycle consistency loss at CycleGAN section 3.2
            self.criterion_identity = nn.L1Loss()       # Identity loss at CycleGAN section 5.2
            # Optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_S.parameters(), self.G_T.parameters()),
                                                lr=params.lr, betas=(params.beta, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_S.parameters(), self.D_T.parameters()),
                                                lr=params.lr, betas=(params.beta, 0.999))
            # 필요에 따라 LR schedulers 추가 선언

    def set_input(self, input):
        ''' 
            Iteration마다 DataLoader로부터 input을 받아서 unpack
        '''
        self.real_S = Variable(input['S_img'].to(self.device))
        self.real_T = Variable(input['T_img'].to(self.device))

    def set_requires_grad(self, model_list, requires_grad=False):
        """
            불필요한 연산을 줄이기 위해 사용
        """
        if not isinstance(model_list, list):
            model_list = [model_list]
        for model in model_list:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.fake_S = self.G_S(self.real_T)     # G_S(T)
        self.recons_T = self.G_T(self.fake_S)   # G_T(G_S(T))
        self.fake_T = self.G_T(self.real_S)     # G_T(S)
        self.recons_S = self.G_S(self.fake_T)   # G_S(G_T(S))

    def train_D(self):
        '''
            real_T -- G_T --> fake_S -->
                                            D_S
                              real_S -->
            이 때 fake_S에서 detach를 해주지 않을 경우 real_T까지 gradient를 계산
        '''
        # D_S training
        pred_real = self.D_S(self.real_S)
        loss_S_real = self.criterion_GAN(pred_real, self.real)   # 진짜 이미지를 진짜라고

        fake_S = self.save_fake_S.query(self.fake_S)
        pred_fake = self.D_S(fake_S.detach())
        loss_S_fake = self.criterion_GAN(pred_fake, self.fake)  # 가짜 이미지를 가짜라고

        self.loss_D_S = (loss_S_fake + loss_S_real)*0.5
        self.loss_D_S.backward()

        # D_T training
        pred_real = self.D_T(self.real_T)
        loss_T_real = self.criterion_GAN(pred_real, self.real)

        fake_T = self.save_fake_T.query(self.fake_T)
        pred_fake = self.D_T(fake_T.detach())
        loss_T_fake = self.criterion_GAN(pred_fake, self.fake)

        self.loss_D_T = (loss_T_fake + loss_T_real)*0.5
        self.loss_D_T.backward()

    def train_G(self):
        lambda_cycle = self.params.lambda_cycle
        lambda_ident = self.params.lambda_ident

        # Identity training
        self.ident_S = self.G_S(self.real_S)
        self.loss_ident_S = self.criterion_identity(self.ident_S, self.real_S)*lambda_ident
        self.ident_T = self.G_T(self.real_T)
        self.loss_ident_T = self.criterion_identity(self.ident_T, self.real_T)*lambda_ident

        # Adversarial training
        self.loss_G_S = self.criterion_GAN(self.D_S(self.fake_S), self.real)
        self.loss_G_T = self.criterion_GAN(self.D_T(self.fake_T), self.real)

        # Cycle consistency training
        self.loss_cycle_S = self.criterion_cycle(self.recons_S, self.real_S)*lambda_cycle
        self.loss_cycle_T = self.criterion_cycle(self.recons_T, self.real_T)*lambda_cycle

        self.loss_G = self.loss_ident_S + self.loss_G_S + self.loss_cycle_S +\
                      self.loss_ident_T + self.loss_G_T + self.loss_cycle_T
        self.loss_G.backward()

    def train(self):
        self.forward()
        # Generator학습에 Discriminator gradient는 필요 x
        self.set_requires_grad([self.D_S, self.D_T], False)
        self.optimizer_G.zero_grad()
        self.train_G()
        self.optimizer_G.step()
        # set_require_grad를 사용하지 않고 train 함수 내에서 detach
        self.set_requires_grad([self.D_S, self.D_T], True)
        self.optimizer_D.zero_grad()
        self.train_D()
        self.optimizer_D.step()

    def get_data_for_logging(self):
        log_for_term = {'G_total': self.loss_G, 'D_total': self.loss_D_S+self.loss_D_T}                   # Terminal에 logging할 정보

        loss_log = {'G_total': self.loss_G, 'G_adversarial': self.loss_G_S+self.loss_G_T,                 # Visdom에 visualize할 loss graph
                    'G_identity': self.loss_ident_S+self.loss_ident_T,
                    'G_cycle':self.loss_cycle_S + self.loss_cycle_T,
                    'D_total': self.loss_D_S + self.loss_D_T}

        img_log = { 'real_S':self.real_S[0], 'real_T':self.real_T[0], 'fake_S':self.fake_S[0],            # Visdom에 visualize할 images
                    'fake_T':self.fake_T[0], 'recons_S':self.recons_S[0], 'recons_T':self.recons_T[0]}

        return log_for_term, loss_log, img_log

    def save(self, ckp_name):
        path = os.path.join(self.params.checkpoint_dir, ckp_name)
        checkpoint = {   'G_S': self.G_S.state_dict(),
                         'G_T': self.G_T.state_dict(),
                         'D_S': self.D_S.state_dict(),
                         'D_T': self.D_T.state_dict(),
                         'G_optimizer': self.optimizer_G.state_dict(),
                         'D_optimizer': self.optimizer_D.state_dict()    }
        torch.save(checkpoint, path)

    def load(self, ckp_path):
        checkpoint = torch.load(ckp_path)
        self.G_S.load_state_dict(checkpoint['G_S'])
        self.G_T.load_state_dict(checkpoint['G_T'])
        self.D_S.load_state_dict(checkpoint['D_S'])
        self.D_T.load_state_dict(checkpoint['D_T'])
        self.optimizer_G.load_state_dict(checkpoint['G_optimizer'])
        self.optimizer_D.load_state_dict(checkpoint['D_optimizer'])











