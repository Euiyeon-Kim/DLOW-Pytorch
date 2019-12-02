import os
import sys
import yaml
import time

import torch

from data import DataLoader
from util.Logger import Logger
from util.utils import*
from model.CycleGAN import CycleGAN
from model.InterpolationGAN import InterpolationGAN

if __name__ == "__main__":

    # Reading configuration file
    conf = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    train_conf = conf['train']

    # Checkpoint directory
    if not os.path.isdir(train_conf['checkpoint_dir']):
        os.mkdir(train_conf['checkpoint_dir'])

    # GPU 할당
    os.environ["CUDA_VISIBLE_DEVICES"] = train_conf['gpu_id']

    # Model 선언
    model = InterpolationGAN(train_conf)
    print_network(model)
    model = torch.nn.DataParallel(model, output_device=1)
    model.cuda()


    # 데이터 로딩
    sys.stdout.write("Loading the data...")
    dataloaders = DataLoader.get_dataloaders(['train', 'val'], train_conf)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    sys.stdout.write(" -done")
    sys.stdout.flush()

    # Logging과 Visualizing
    num_batches = len(train_dl)
    logger = Logger(train_conf)

    # Train
    train_conf['total_iter'] = num_batches * train_conf['num_epochs']
    for epoch in range(train_conf['start_epoch'], train_conf['start_epoch'] + train_conf['num_epochs']):
        
        for i, batch in enumerate(train_dl):
            # Epoch가 증가해도 초기화 되지 않음
            train_conf['cur_iter'] += 1
            # Actual training
            model.module.set_input(batch)
            model.module.train()
            
            # Logging on terminal
            sys.stdout.write('\nEpoch %03d/%03d [%04d/%04d] %d'%(epoch, train_conf['num_epochs'], i+1, num_batches, train_conf['cur_iter']))
            sys.stdout.flush()

            # Logging on tensorboard
            if i % train_conf['save_summary_steps'] == 0:
                loss_log, img_log = model.module.get_data_for_logging()
                logger.log(loss_log, img_log)
        
        if (epoch % 50) == 0: # validation 성능으로 수정하기
            model.module.save(str(epoch)+'.pth.tar') 
   
