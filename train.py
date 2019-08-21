import argparse
import itertools
import logging
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm # for visualized logging

import utils
from evaluate import evaluate
import data.data_loader as dataLoader
import model.CycleGAN as model

parser = argparse.ArgumentParser()
# Related to directory
parser.add_argument('--S_dir', default='./data/GTA5', help="Where to find Source dataset")
parser.add_argument('--T_dir', default='./data/bdd100k', help="Where to find Target dataset")
parser.add_argument('--model_dir', default='./model', help="Directory containing the params")
parser.add_argument('--checkpoint_dir', default='./model/checkpoint', help="Directory to save model")
parser.add_argument('--restore_file', default=None, help="Name of the file in --checkpoint_dir")
parser.add_argument('--output_dir', default='./output', help="Directory to save outputs")

# Related to learning rate policy
parser.add_argument('--lr_policy', type=str, default='linear', help="Learning rate scheduler")
parser.add_argument('--epoch_count', type=int, default=1, help="Starting epoch") # For continuous training
parser.add_argument('--start_decay', type=int, default=100, help="# of iter at stating learning rate decay")
parser.add_argument('--decay_cycle', type=int, default=100, help="# of iter to linearly decay learning rate")
parser.add_argument('--lr_decay_iters', type=int, default=50, help="Multiply by gamma every lr_decay_iters iterations")


def actual_train(epoch, S2T, T2S, D_S, D_T, ans_R, ans_F, G_optimizer, D_optimizer, losses, dataloader, params):
    S2T.train()
    T2S.train()
    D_S.train()
    D_T.train()

    summary = []
    avg_Gloss = utils.RunningAverage()
    avg_Dloss = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:

        for i, batch in enumerate(dataloader):

            if params.cuda:
                batch = batch.cuda(async=True)

            batch = Variable(batch) # Normalized
            real_S = batch['S_img']
            real_T = batch['T_img']

            ##### Training for Generators #####
            D_S.set_requires_grad(False)
            D_T.set_requires_grad(False)
            G_optimizer.zero_grad()

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
            S2T_GAN_loss = losses['criterion_GAN'](pred_for_fakeS, ans_R)*5.0 # hyper-parameter
            T2S_GAN_loss = losses['criterion_GAN'](pred_for_fakeT, ans_R)*5.0

            # Cycle consistency loss at section 3.2
            recons_S = T2S(fake_T)
            recons_T = S2T(fake_S)
            S_cycle_loss = losses['criterion_cycle'](recons_S, real_S)*10.0 # hyper-parameter(10.0*0.5)
            T_cycle_loss = losses['criterion_cycle'](recons_T, real_T)*10.0

            G_loss = S_identity_loss + T_identity_loss + S2T_GAN_loss + T2S_GAN_loss + S_cycle_loss + T_cycle_loss
            G_loss.backward()

            G_optimizer.step()

            ##### Training for Discriminator #####
            D_S.set_requires_grad(True)
            D_T.set_requires_grad(True)

            D_optimizer.zero_grad()

            pred_for_realS = D_S(real_S, ans_R)
            pred_for_fakeT = D_S(fake_T, ans_F)
            pred_for_reconsS = D_S(recons_S, ans_R)
            DS_loss = pred_for_realS + pred_for_fakeT+pred_for_reconsS
            DS_loss.backward()

            pred_for_realT = D_T(real_T, ans_R)
            pred_for_fakeS = D_T(fake_S, ans_F)
            pred_for_reconsT = D_T(recons_T, ans_R)
            DT_loss = pred_for_realT + pred_for_fakeS + pred_for_reconsT
            DT_loss.backward()

            D_optimizer.step()

            if i % params.save_summary_steps == 0:
                # Extract data from torch variable / Move to CPU / Convert to numpy arrays
                real_S = real_S.data.cpu().numpy()
                real_T = real_T.data.cpu().numpy()
                fake_S = fake_S.data.cpu().numpy()
                fake_T = fake_T.data.cpu().numpy()
                recons_S = recons_S.data.cpu().numpy()
                recons_T = recons_T.data.cpu().numpy()

                # Compute all metrics on this batch
                summary_batch = {}
                summary_batch['G_loss'] = G_loss.item()
                summary_batch['DS_loss'] = DS_loss.item()
                summary_batch['DT_loss'] = DT_loss.item()

                summary.append(summary_batch)

                real_S = np.transpose(real_S[0], (1, 2, 0)).astype('uint8')
                real_T = np.transpose(real_T[0], (1, 2, 0)).astype('uint8')
                fake_S = np.transpose(fake_S[0], (1, 2, 0)).astype('uint8')
                fake_T = np.transpose(fake_T[0], (1, 2, 0)).astype('uint8')
                recons_S = np.transpose(recons_S[0], (1, 2, 0)).astype('uint8')
                recons_T = np.transpose(recons_T[0], (1, 2, 0)).astype('uint8')

                img1 = Image.fromarray(real_S, 'RGB')
                img1.save(os.path.join(args.output_dir, 'Epoch' + str(epoch) + '_Step' + str(i) + '_real_S.png'))
                img2 = Image.fromarray(fake_T, 'RGB')
                img2.save(os.path.join(args.output_dir, 'Epoch' + str(epoch) + '_Step' + str(i) + '_fake_T.png'))
                img3 = Image.fromarray(recons_S, 'RGB')
                img3.save(os.path.join(args.output_dir, 'Epoch' + str(epoch) + '_Step' + str(i) + '_recons_S.png'))
                img4 = Image.fromarray(real_T, 'RGB')
                img4.save(os.path.join(args.output_dir, 'Epoch' + str(epoch) + '_Step' + str(i) + '_real_T.png'))
                img5 = Image.fromarray(fake_S, 'RGB')
                img5.save(os.path.join(args.output_dir, 'Epoch' + str(epoch) + '_Step' + str(i) + '_fake_S.png'))
                img6 = Image.fromarray(recons_T, 'RGB')
                img6.save(os.path.join(args.output_dir, 'Epoch' + str(epoch) + '_Step' + str(i) + '_recons_T.png'))

            avg_Gloss.update(G_loss.item())
            avg_Dloss.update(DS_loss.item()+DT_loss.item())
            t.set_postfix(G_loss='{:05.3f}'.format(avg_Gloss()))
            t.set_postfix(D_loss='{:05.3f}'.format(avg_Dloss()))
            t.update()


def train(S2T, T2S, D_S, D_T, ans_R, ans_F, train_dataloader, val_dataloader, G_optimizer, D_optimizer, losses,
          params, model_dir, checkpoint_dir, restore_file=None):
    # Restoring model
    if args.restore_file is not None:
        restore_path = os.path.join(args.checkpoint_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, S2T, T2S, D_S, D_T)

    best_val_error = np.inf

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        actual_train(epoch, S2T, T2S, D_S, D_T, ans_R, ans_F, G_optimizer, D_optimizer,
                     losses, train_dataloader, params)

        val_losses = evaluate(S2T, T2S, D_S, D_T, ans_R, ans_F, losses, dataloaders, params)

        val_error = val_losses['Total_loss']
        is_best = val_error <= best_val_error

        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        utils.save_checkpoint({'epoch': epoch+1,
                               'S2T_state_dict': S2T.state_dict(),
                               'T2S_state_dict': T2S.state_dict(),
                               'D_S_state_dict': D_S.state_dict(),
                               'D_T_state_dict': D_T.state_dict(),
                               'G_optimizer_state_dict': G_optimizer.state_dict(),
                               'D_optimizer_state_dict': D_optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint_path=checkpoint_dir)

        if is_best:
            logging.info("< - Found New Best Parameters - >")
            best_val_error = val_error

            best_json_path = os.path.join(model_dir, "Test_best_results.json")
            utils.save_dict_to_json(val_losses, best_json_path)

        last_json_path = os.path.join(model_dir, "Test_last_results.json")
        utils.save_dict_to_json(val_losses, last_json_path)



if __name__=='__main__':
    args = parser.parse_args()

    # Hyper-parameters setting
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()
    torch.manual_seed(1385)

    # Logger setting
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Dataloader setting
    logging.info("Loading the data...")
    dataloaders = dataLoader.get_dataloaders(['train', 'val'], args.S_dir, args.T_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    logging.info("- done")

    # Define model and initialization
    S2T = model.Generator(params.S_nc, params.T_nc)
    T2S = model.Generator(params.T_nc, params.S_nc)
    D_S = model.Discriminator(params.S_nc)
    D_T = model.Discriminator(params.T_nc)

    if params.cuda:
        S2T.cuda()
        T2S.cuda()
        D_S.cuda()
        D_T.cuda()

    utils.init_weights(S2T)
    utils.init_weights(T2S)
    utils.init_weights(D_S)
    utils.init_weights(D_T)

    # Define Lossess
    losses = {}
    losses['criterion_GAN'] = nn.MSELoss() #
    losses['criterion_cycle'] = nn.L1Loss() # C
    losses['criterion_identity'] = nn.L1Loss() #

    # Define optimizers
    G_optimizer = optim.Adam(itertools.chain(S2T.parameters(), T2S.parameters()),
                             lr=params.learning_rate, betas=(params.beta, 0.999))
    D_optimizer = optim.Adam(itertools.chain(D_S.parameters(), D_T.parameters()),
                             lr=params.learning_rate, betas=(params.beta, 0.999))

    # Define answer for discriminator
    Tensor = torch.cuda.FloatTensor if params.cuda else torch.Tensor
    ans_R = Variable(Tensor(params.batch_size).fill_(1.0), requires_grad=False)
    ans_F = Variable(Tensor(params.batch_size).fill_(0.0), requires_grad=False)

    # Define learning rate scheduler if nesessary
    # lr_scheduler_G = utils.get_scheduler(G_optimizer, args)
    # lr_scheduler_D = utils.get_scheduler(D_optimizer, args)

    logging.info("Start train for {}epochs".format(params.num_epochs))
    train(S2T, T2S, D_S, D_T, ans_R, ans_F, train_dl, val_dl, G_optimizer, D_optimizer, losses, params,
          args.model_dir, args.checkpoint_dir, args.restore_file)



