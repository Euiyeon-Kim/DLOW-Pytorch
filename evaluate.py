import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.CycleGAN as model
import data.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/Kitti', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='./model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="Name of the file in --model_dir/checkpoint containing weights to load")


def evaluate(S2T, T2S, D_S, D_T, ans_R, ans_F, losses, dataloader, params):

    S2T.eval() # Set model to evaluation mode
    T2S.eval()
    D_S.eval()
    D_T.eval()

    summary = [] # Summary for current evaluation loop

    for batch in dataloader:
        if params.cuda:
            batch = batch.cuda(async=True)

        batch = Variable(batch)# torch Variable로 변환
        real_S = batch['S_img']
        real_T = batch['T_img']

        # Identity loss at section 5.2
        same_T = S2T(real_T)
        same_S = S2T(real_S)
        S_identity_loss = losses['criterion_identity'](same_S, real_S)
        T_identity_loss = losses['criterion_identity'](same_T, real_T)

        # Adversarial loss at section 3.1
        fake_T = S2T(real_S)
        fake_S = T2S(real_T)
        pred_for_fakeS = D_S(fake_S)
        pred_for_fakeT = D_T(fake_T)
        S2T_GAN_loss = losses['criterion_GAN'](pred_for_fakeS, ans_R) * 5.0  # hyper-parameter
        T2S_GAN_loss = losses['criterion_GAN'](pred_for_fakeT, ans_R) * 5.0

        # Cycle consistency loss at section 3.2
        recons_S = T2S(fake_T)
        recons_T = S2T(fake_S)
        S_cycle_loss = losses['criterion_cycle'](recons_S, real_S) * 10.0  # hyper-parameter(10.0*0.5)
        T_cycle_loss = losses['criterion_cycle'](recons_T, real_T) * 10.0

        G_loss = S_identity_loss + T_identity_loss + S2T_GAN_loss + T2S_GAN_loss + S_cycle_loss + T_cycle_loss

        pred_for_realS = D_S(real_S, ans_R)
        pred_for_fakeT = D_S(fake_T, ans_F)
        pred_for_reconsS = D_S(recons_S, ans_R)
        DS_loss = pred_for_realS + pred_for_fakeT + pred_for_reconsS

        pred_for_realT = D_T(real_T, ans_R)
        pred_for_fakeS = D_T(fake_S, ans_F)
        pred_for_reconsT = D_T(recons_T, ans_R)
        DT_loss = pred_for_realT + pred_for_fakeS + pred_for_reconsT

        summary_batch = {}
        summary_batch['G_loss'] = G_loss.item()
        summary_batch['DS_loss'] = DS_loss.item()
        summary_batch['DT_loss'] = DT_loss.item()
        summary_batch['Total_loss'] = DS_loss.item() + DT_loss.item() + G_loss.item()
        summary.append(summary_batch)


    metrics_mean = {metric: np.mean([x[metric] for x in summary]) for metric in summary[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


