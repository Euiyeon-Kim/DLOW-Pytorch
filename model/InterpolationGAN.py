import os
import sys
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import BaseNetwork as Base
from util import utils

sys.path.append("..")

class InterpolationGAN(nn.Module):

    def __init__(self, conf, is_train=True):
        super(InterpolationGAN, self).__init__()
        self.conf = conf
        
        # Generator 생성 및 초기화
        self.G_S = Base.Stoch_Generator(conf['nlatent'], conf['T_nc'], conf['S_nc'], conf['ngf'], 
                                        conf['use_dropout'], conf['n_res_blocks'])                  # T를 S로 변환하는 Generator 
        self.G_T = Base.Stoch_Generator(conf['nlatent'], conf['S_nc'], conf['T_nc'], conf['ngf'], 
                                        conf['use_dropout'], conf['n_res_blocks'])                  # S를 T로 변환하는 Generator

        # Domainess encoded latent vector Generator 생성 및 초기화
        self.G_D = nn.Linear(1, conf['nlatent'])
        utils.init_weights(self.G_D)

        # Discriminator 생성 및 초기화
        self.D_S = Base.Discriminator(conf['S_nc'], conf['ndf']) # domain S를 구분하는 Discriminator
        self.D_T = Base.Discriminator(conf['T_nc'], conf['ndf']) # domain T를 구분하는 Discriminator

        # Criterion 및 Optimizer 생성
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_D.parameters(), self.G_S.parameters(), self.G_T.parameters()),
                                              lr=conf['lr'], betas=(conf['beta'], 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_S.parameters(), self.D_T.parameters()),
                                              lr=conf['lr'], betas=(conf['beta'], 0.999))
        self.criterionGAN = nn.MSELoss(reduction='mean')
        self.criterion_cycle = nn.L1Loss(reduction='mean')

        # Answer for discriminator
        self.ans_real = Variable(torch.ones([self.conf['batch_size'], 1, 23, 23]), requires_grad=False).cuda()
        self.ans_fake = Variable(torch.zeros([self.conf['batch_size'], 1, 23, 23]), requires_grad=False).cuda()

    def set_input(self, input):
        ''' 
            Iteration마다 DataLoader로부터 input을 받아서 unpack
            size : batch, channel, height, width
        '''
        self.real_S = Variable(input['S_img']).cuda()
        self.real_T = Variable(input['T_img']).cuda()
        self.domainess = utils.get_domainess(self.conf['cur_iter'], self.conf['total_iter'], 1).cuda()
        
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

    def train(self):
        # Make flow S to T
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        
        self.Z = torch.unsqueeze(torch.unsqueeze(self.G_D(self.domainess), 2), 3)              # domainess Z (1, 16, 1, 1)
        self.fake_T = self.G_T(self.real_S, self.Z)                                            # S에서 T쪽으로 z만큼 이동
        self.recons_S = self.G_S(self.fake_T, self.Z)                                          # fake_T에서 다시 S쪽으로 z만큼 이동 
        
        # Train G
        self.loss_cycle_S = self.criterion_cycle(self.recons_S, self.real_S)*self.conf['lambda_cycle']
        self.loss_G_T_D_S = self.criterionGAN(self.D_S(self.fake_T), self.ans_real)
        self.loss_G_T_D_T = self.criterionGAN(self.D_T(self.fake_T), self.ans_real)
        self.loss_G_S2T = (1 - self.domainess)*self.loss_G_T_D_S + self.domainess*self.loss_G_T_D_T + self.loss_cycle_S
        self.loss_G_S2T.backward()
        self.optimizer_G.step()

        # Train D
        loss_S_real_S = self.criterionGAN(self.D_S(self.real_S), self.ans_real)                # Real S images
        loss_S_fake_T = self.criterionGAN(self.D_S(self.fake_T.detach()), self.ans_fake)       # S에서 T방향으로 z만큼 이동
        self.loss_D_S_G_T = loss_S_real_S + loss_S_fake_T
        
        loss_T_real_T = self.criterionGAN(self.D_T(self.real_T), self.ans_real)                # Real T images
        loss_T_fake_T = self.criterionGAN(self.D_T(self.fake_T.detach()), self.ans_fake)       # S에서 T방향으로 z만큼 이동
        self.loss_D_T_G_T = loss_T_real_T + loss_T_fake_T
        
        self.loss_D_S2T = (1 - self.domainess)*self.loss_D_S_G_T + self.domainess*self.loss_D_T_G_T
        self.loss_D_S2T.backward
        self.optimizer_D.step()

        torch.cuda.empty_cache()

        # Make flow T to S
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        self.Z_1 = torch.unsqueeze(torch.unsqueeze(self.G_D(1-self.domainess), 2), 3)          # domainess 1 - Z (1, 16, 1, 1)
        self.fake_S = self.G_S(self.real_T, self.Z_1)                                          # T에서 S쪽으로 1-z만큼 이동
        self.recons_T = self.G_T(self.fake_S, self.Z_1)                                        # fake_S에서 다시 T쪽으로 1-z만큼 이동

        # Train G
        self.loss_cycle_T = self.criterion_cycle(self.recons_T, self.real_T)*self.conf['lambda_cycle']
        self.loss_G_S_D_S = self.criterionGAN(self.D_S(self.fake_S), self.ans_real)
        self.loss_G_S_D_T = self.criterionGAN(self.D_T(self.fake_S), self.ans_real)
        self.loss_G_T2S = self.domainess*self.loss_G_S_D_S + (1-self.domainess)*self.loss_G_S_D_T + self.loss_cycle_T
        self.loss_G_T2S.backward()
        self.optimizer_G.step()

        # Train D
        loss_S_real_S = self.criterionGAN(self.D_S(self.real_S), self.ans_real)                # Real S images
        loss_S_fake_S = self.criterionGAN(self.D_S(self.fake_S.detach()), self.ans_fake)       # T에서 S방향으로 1-z만큼 이동
        self.loss_D_S_G_S = loss_S_real_S + loss_S_fake_S

        loss_T_real_T = self.criterionGAN(self.D_T(self.real_T), self.ans_real)                # Real T images
        loss_T_fake_S = self.criterionGAN(self.D_T(self.fake_S.detach()), self.ans_fake)       # T에서 S방향으로 1-z만큼 이동
        self.loss_D_T_G_S = loss_T_real_T + loss_T_fake_S
        
        self.loss_D_T2S = self.domainess*self.loss_D_S_G_S + (1-self.domainess)*self.loss_D_T_G_S
        self.loss_D_T2S.backward()
        self.optimizer_D.step()

        torch.cuda.empty_cache()

    def get_data_for_logging(self): # Retun할 때 data.cpu로 옮겨서 return하기.

        loss_log = {'G_total': self.loss_G_T2S + self.loss_G_S2T, # Visdom에 visualize할 loss graph
                    'G_adversarial': self.domainess*self.loss_G_S_D_S + (1-self.domainess)*self.loss_G_S_D_T + (1 - self.domainess)*self.loss_G_T_D_S + self.domainess*self.loss_G_T_D_T, 
                    'G_cycle':self.loss_cycle_S + self.loss_cycle_T,
                    'D_total': self.loss_D_T2S + self.loss_D_S2T }

        img_log = { 'real_S':self.real_S[0], 'real_T':self.real_T[0], 'fake_S':self.fake_S[0], # Visdom에 visualize할 images
                    'fake_T':self.fake_T[0], 'recons_S':self.recons_S[0], 'recons_T':self.recons_T[0]}

        return loss_log, img_log

    def save(self, ckp_name):
        path = os.path.join(self.conf['checkpoint_dir'], ckp_name)
        checkpoint = { 'G_D': self.G_D.state_dict(),
                       'G_S': self.G_S.state_dict(),
                       'G_T': self.G_T.state_dict(),
                       'D_S': self.D_S.state_dict(),
                       'D_T': self.D_T.state_dict(),
                       'G_optimizer': self.optimizer_G.state_dict(),
                       'D_optimizer': self.optimizer_D.state_dict() }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.G_D.load_state_dict(checkpoint['G_D'])
        self.G_S.load_state_dict(checkpoint['G_S'])
        self.G_T.load_state_dict(checkpoint['G_T'])
        self.D_S.load_state_dict(checkpoint['D_S'])
        self.D_T.load_state_dict(checkpoint['D_T'])
        self.optimizer_G.load_state_dict(checkpoint['G_optimizer'])
        self.optimizer_D.load_state_dict(checkpoint['D_optimizer']) 