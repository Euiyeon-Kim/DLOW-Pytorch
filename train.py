#!/usr/bin/python3

import argparse
import logging
import torch

from data import DataLoader
from model.Interpolation import InterpolationGAN

from util.utils import Logger


parser = argparse.ArgumentParser()

# Basic settings
parser.add_argument('--cuda', type=bool, default=True, help="Use GPU computation")
parser.add_argument('--gpu_id', type=int, default=0, help="GPU id to use")
parser.add_argument('--start_epoch', type=int, default=0, help="Start point to train")
parser.add_argument('--num_epochs', type=int, default=100, help="# of epoch to train")
parser.add_argument('--num_worker', type=int, default=4, help="# of cpu threads to use during batch generation")
parser.add_argument('--save_summary_steps', type=int, default=100, help="# of iter to save current status")
parser.add_argument('--buf_size', type=int, default=50, help='Buffer size to save previous generated images')

# Related to dataset
parser.add_argument('--root_dir', type=str, default='./dataset', help="Where to find dataset")
parser.add_argument('--S_nc', type=int, default=3, help="Source dataset's channels")
parser.add_argument('--T_nc', type=int, default=3, help="Target dataset's channels")
parser.add_argument('--resize_W', type=int, default=1024, help='Resize data to have this width') # 아직 안쓰는 중
parser.add_argument('--resize_H', type=int, default=576, help='Resize data to have this height') # 아직 안쓰는 중
parser.add_argument('--fixed_pair', type=bool, default=True, help='Maintain Source dataset and Target dataset\'s pair')

# Related to directory
parser.add_argument('--checkpoint_dir', default='./model/checkpoint', help="Directory to save model")
parser.add_argument('--restore_filename', default=None, help="Name of the file in --checkpoint_dir")
parser.add_argument('--output_dir', default='./output', help="Directory to save outputs")

# Related to training (Hyper-parameters)
parser.add_argument('--lr', type=float, default=0.0002, help="Learning rate")
parser.add_argument('--beta', type=float, default=0.5, help="Used with Adam optimizer")
parser.add_argument('--n_res_blocks', type=int, default=9, help="Number of residual blocks used to make G")
parser.add_argument('--lamda_cycle', type=float, default=10, help="Lamda for cycle consistency loss")
parser.add_argument('--lamda_ident', type=float, default=10, help="Lamda for identity loss")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
parser.add_argument('--use_dropout', type=bool, default=False, help="Wether to use dropout or not")

# Related to learning rate policy
parser.add_argument('--lr_policy', type=str, default='linear', help="Learning rate scheduler")
parser.add_argument('--start_decay', type=int, default=100, help="# of iter at stating learning rate decay")
parser.add_argument('--decay_cycle', type=int, default=100, help="# of iter to linearly decay learning rate")
parser.add_argument('--lr_decay_iters', type=int, default=50, help="Multiply by gamma every lr_decay_iters iterations")


if __name__ == "__main__":

    params = parser.parse_args()
    
    # GPU 사용 가능 여부 확인
    params.cuda = torch.cuda.is_available()

    InterpolationGAN(params)

    logging.info("Loading the data...")
    dataloaders = DataLoader.get_dataloaders(['train', 'val'], params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    logging.info(" -done")

    # Logging 을 어예할지 잘 생각해보자꾸나.
    #logger = Logger(params.n_epochs, len(dataloaders))

    ###### Training ######
    for epoch in range(params.epoch, params.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ###### Generators A2B and B2A ######
            paramsimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            paramsimizer_G.step()
            ###################################

            ###### Discriminator A ######
            paramsimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            paramsimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            paramsimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            paramsimizer_D_B.step()
            ###################################

            # Progress report (http://localhost:8097)
            logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                        'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                        images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B.pth')
    ###################################
