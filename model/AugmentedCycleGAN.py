import os
import sys
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import BaseNetwork as Base
from util import utils

sys.path.append("..")

class AugmentedCycleGAN(nn.Module):

    def __init__(self, params, is_train=True):
        super(AugmentedCycleGAN, self).__init__()
        self.params = params
        
        # Device 설정
        if params.cuda: # 후에 multi-GPU coding할 수 있으면 적용
            self.device = torch.device('cuda:{}'.format(params.gpu_id)) 
        else:
            self.device = torch.device('cpu')

        # Generator 생성 및 초기화
        self.G_S = Base.Stoch_Generator(params.nlatent, params.T_nc, params.S_nc, params.ngf, 
                                        params.use_dropout, params.n_res_blocks, self.device)  # T를 S로 변환하는 Generator                                 
        self.G_T = Base.Stoch_Generator(params.nlatent, params.S_nc, params.T_nc, params.ngf, 
                                        params.use_dropout, params.n_res_blocks, self.device)  # S를 T로 변환하는 Generator

        # Domainess encoded latent vector Generator 생성 및 초기화
        self.G_D = nn.Linear(1, params.nlatent)
        self.G_D.to(self.device)
        utils.init_weights(self.G_D)

        # Discriminator 생성 및 초기화
        self.D_S = Base.Discriminator(params.S_nc, params.ndf, self.device)  # domain S를 구분하는 Discriminator
        self.D_T = Base.Discriminator(params.T_nc, params.ndf, self.device)  # domain T를 구분하는 Discriminator

        # Criterion 및 Optimizer 생성
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_D.parameters(), self.G_S.parameters(), self.G_T.parameters()),
                                                lr=params.lr, betas=(params.beta, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_S.parameters(), self.D_T.parameters), lr=params.lr, betas=(params.beta, 0.999))
        self.criterionGAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()

    def set_input(self, input):
        ''' 
            Iteration마다 DataLoader로부터 input을 받아서 unpack
            size : batch, channel, height, width
        '''
        self.real_S = Variable(input['S_img'].to(self.device))
        self.real_T = Variable(input['T_img'].to(self.device))
        self.domainess = utils.get_domainess(self.params.cur_iter, self.params.total_iter, self.params.batch_size)

        # Answer for discriminator
        self.ans_real = Variable(torch.ones([self.params.batch_size, 1, 16, 23]).to(self.device), requires_grad=False)
        self.ans_fake = Variable(torch.zeros([self.params.batch_size, 1, 16, 23]).to(self.device), requires_grad=False)

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
        self.Z = torch.unsqueeze(torch.unsqueeze(self.G_D(self.domainess), 2), 3).to(self.device)      # domainess Z
        self.Z_1 = torch.unsqueeze(torch.unsqueeze(self.G_D(1-self.domainess), 2), 3).to(self.device)  # domainess 1 - Z
        self.fake_T = self.G_T(self.real_S, self.Z)                                                    # S에서 T쪽으로 z만큼 이동
        self.recons_S = self.G_S(self.fake_T, self.Z)                                                  # fake_T에서 다시 S쪽으로 z만큼 이동 
        self.fake_S = self.G_S(self.real_T, self.Z_1)                                                  # T에서 S쪽으로 1-z만큼 이동
        self.recons_T = self.G_T(self.fake_S, self.Z_1)                                                # fake_S에서 다시 T쪽으로 1-z만큼 이동

    def train(self):
        lambda_cycle = self.params.lambda_cycle

        self.forward()
        
        # Make flow S to T
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        
        D_S_pred_real_S = self.D_S(self.real_S)                             # Real S images
        loss_S_real_S = self.criterion_GAN(D_S_pred_real_S, self.ans_real)    
        D_S_pred_fake_T = self.D_S(self.fake_T.detach())                    # S에서 T방향으로 z만큼 이동
        loss_S_fake_T = self.criterionGAN(D_S_pred_fake_T, self.ans_fake)
        D_T_pred_real_T = self.D_T(self.real_T)                             # Real T images
        loss_T_real_T = self.criterion_GAN(D_T_pred_real_T, self.ans_real)     
        D_T_pred_fake_T = self.D_T(fake_T.detach())                         # S에서 T방향으로 z만큼 이동
        loss_T_fake_T = self.criterion_GAN(D_T_pred_fake_T, self.ans_fake)
        self.loss_D_S_G_T = loss_S_real_S + loss_S_fake_T
        self.loss_D_T_G_T = loss_T_real_T + loss_T_fake_T
        self.loss_D_S2T = (1 - self.domainess)*self.loss_D_S_G_T + self.domainess*self.loss_D_T_G_T
        
        self.loss_cycle_S = self.criterion_cycle(self.recons_S, self.real_S)*lambda_cycle
        self.loss_G_T_D_S = self.criterion_GAN(self.D_S(self.fake_T), self.real)
        self.loss_G_T_D_T = self.criterionGAN(self.D_T(self.fake_T), self.real)
        self.loss_G_S2T = (1 - self.domainess)*self.loss_G_T_D_S + self.domainess*self.loss_G_T_D_T + self.loss_cycle_S
        self.optimizer_G.step()
        self.optimizer_D.step()

        # Make flow T to S
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        
        D_S_pred_real_S = self.D_S(self.real_S)                             # Real S images
        loss_S_real_S = self.criterion_GAN(D_S_pred_real_S, self.ans_real)    
        D_S_pred_fake_S = self.D_S(self.fake_S.detach())                    # T에서 S방향으로 1-z만큼 이동
        loss_S_fake_S = self.criterionGAN(D_S_pred_fake_S, self.ans_fake)
        D_T_pred_real_T = self.D_T(self.real_T)                             # Real T images
        loss_T_real_T = self.criterion_GAN(D_T_pred_real_T, self.ans_real)     
        D_T_pred_fake_S = self.D_T(fake_S.detach())                         # T에서 S방향으로 1-z만큼 이동
        loss_T_fake_S = self.criterion_GAN(D_T_pred_fake_S, self.ans_fake)
        self.loss_D_S_G_S = loss_S_real_S + loss_S_fake_S
        self.loss_D_T_G_S = loss_T_real_T + loss_T_fake_S
        self.loss_D_T2S = self.domainess*self.loss_D_S_G_S + (1-self.domainess)*self.loss_D_T_G_S
        
        self.loss_cycle_T = self.criterion_cycle(self.recons_T, self.real_T)*lambda_cycle
        self.loss_G_S_D_S = self.criterion_GAN(self.D_S(self.fake_S), self.real)
        self.loss_G_S_D_T = self.criterionGAN(self.D_T(self.fake_S), self.real)
        self.loss_G_T2S = self.domainess*self.loss_G_S_D_S + (1-self.domainess)*self.loss_G_S_D_T + self.loss_cycle_S
        self.optimizer_G.step()
        self.optimizer_D.step()

    def get_data_for_logging(self):
        log_for_term = {'G_total': self.loss_G_T2S + self.loss_G_S2T, 'D_total': self.loss_D_T2S+self.loss_D_S2T}  # Terminal에 logging할 정보

        loss_log = {'G_total': self.loss_G_T2S + self.loss_G_S2T,                                                  # Visdom에 visualize할 loss graph
                    'G_adversarial': self.domainess*self.loss_G_S_D_S + (1-self.domainess)*self.loss_G_S_D_T + (1 - self.domainess)*self.loss_G_T_D_S + self.domainess*self.loss_G_T_D_T,                   
                    'G_cycle':self.loss_cycle_S + self.loss_cycle_T,
                    'D_total': self.loss_D_T2S + self.loss_D_S2T }

        img_log = { 'real_S':self.real_S[0], 'real_T':self.real_T[0], 'fake_S':self.fake_S[0],                     # Visdom에 visualize할 images
                    'fake_T':self.fake_T[0], 'recons_S':self.recons_S[0], 'recons_T':self.recons_T[0]}

        return log_for_term, loss_log, img_log

    def save(self, ckp_name):
        path = os.path.join(self.params.checkpoint_dir, ckp_name)
        checkpoint = {   'G_D': self.G_D.state_dict(),
                         'G_S': self.G_S.state_dict(),
                         'G_T': self.G_T.state_dict(),
                         'D_S': self.D_S.state_dict(),
                         'D_T': self.D_T.state_dict(),
                         'G_optimizer': self.optimizer_G.state_dict(),
                         'D_optimizer': self.optimizer_D.state_dict()    }
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

