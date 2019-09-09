import argparse
import time
import sys

import torch

from data import DataLoader
from util.Logger import Logger
from model.Interpolation import InterpolationGAN

parser = argparse.ArgumentParser()

# Basic settings
parser.add_argument('--cuda', type=bool, default=True, help="Use GPU computation")
parser.add_argument('--gpu_id', type=int, default=0, help="GPU id to use")
parser.add_argument('--start_epoch', type=int, default=0, help="Start point to train")
parser.add_argument('--num_epochs', type=int, default=100, help="# of epoch to train")
parser.add_argument('--num_workers', type=int, default=4, help="# of cpu threads to use during batch generation")
parser.add_argument('--save_summary_steps', type=int, default=100, help="# of iter to save current status")
parser.add_argument('--buf_size', type=int, default=50, help="Buffer size to save previous generated images")
parser.add_argument('--image_saving_step', type=int, default=10, help="Save results on every # iters")

# Related to dataset
parser.add_argument('--root_dir', type=str, default='/mnt/hdd0/datasets/DLOW', help="Where to find dataset")
parser.add_argument('--S_nc', type=int, default=3, help="Source dataset's channels")
parser.add_argument('--T_nc', type=int, default=3, help="Target dataset's channels")
parser.add_argument('--resize_W', type=int, default=400, help='Resize data to have this width')
parser.add_argument('--resize_H', type=int, default=300, help='Resize data to have this height')
parser.add_argument('--fixed_pair', type=bool, default=True, help='Maintain Source dataset and Target dataset\'s pair')

# Related to directory
parser.add_argument('--checkpoint_dir', default='./model/checkpoint', help="Directory to save model")
parser.add_argument('--restore_filename', default=None, help="Name of the file in --checkpoint_dir")
parser.add_argument('--output_dir', default='/mnt/hdd0/outputs/DLOW', help="Directory to save outputs")

# Related to training (Hyper-parameters)
parser.add_argument('--lr', type=float, default=0.0002, help="Learning rate")
parser.add_argument('--beta', type=float, default=0.5, help="Used with Adam optimizer")
parser.add_argument('--n_res_blocks', type=int, default=9, help="Number of residual blocks used to make G")
parser.add_argument('--lambda_cycle', type=float, default=10, help="Lambda for cycle consistency loss")
parser.add_argument('--lambda_ident', type=float, default=10, help="Lambda for identity loss")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
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

    # Model 선언
    interpolationGAN = InterpolationGAN(params)

    # 데이터 로딩
    sys.stdout.write("Loading the data...")
    dataloaders = DataLoader.get_dataloaders(['train', 'val'], params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    sys.stdout.write(" -done")
    sys.stdout.flush()

    # Logging과 Visualizing
    logger = Logger(params, len(train_dl))

    total_iter = 0                              # Total number of training iteration
    # Train
    for epoch in range(params.start_epoch, params.start_epoch + params.num_epochs):
        epoch_start_time = time.time()          # How long does it takes for 1 epoch
        cur_iter = 0                            # Current iter in current epoch

        for i, batch in enumerate(train_dl):
            iter_start_time = time.time()       # How long does it takes for 1 iter

            # Actual training
            interpolationGAN.set_input(batch)
            interpolationGAN.train()
            term_log, loss_log, img_log = interpolationGAN.get_data_for_logging()
            logger.log(term_log, loss_log, img_log)























