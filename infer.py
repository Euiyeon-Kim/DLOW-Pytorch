import os
import sys
import yaml
import torch
import random

from data import DLOWDataLoader
from util.Logger import Logger
from model.InterpolationGAN import InterpolationGAN


if __name__ == "__main__":
    # Reading configuration file
    conf = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    infer_conf = conf['infer']

    # Checkpoint directory
    if not os.path.exists(os.path.join(infer_conf['checkpoint_dir'], "last.pth.tar")):
        print(os.path.join(infer_conf['checkpoint_dir'], "last.pth.tar"))
        print("Checkpoint file doesn't exists")
        exit()
    if not os.path.isdir(infer_conf['DLOW_dir']):
        os.mkdir(infer_conf['DLOW_dir'])

    # Device 할당
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = infer_conf['gpu_id']
        device = torch.device('cuda')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device('cpu')
    print("Using ", device)

    # Model parameter loading
    model = InterpolationGAN(infer_conf, device=device, is_train=False)
    print(model)
    model = torch.nn.DataParallel(model, output_device=1)
    model.to(device)
    model.module.load(os.path.join(infer_conf['checkpoint_dir'], "last.pth.tar"))

    for param in model.parameters():
        param.requires_grad = False
    
    dataloaders = DLOWDataLoader.get_DLOW_dataloaders(['train'], infer_conf, False)
    train_dl = dataloaders['train']

    num_batches = len(train_dl)
    logger = Logger(infer_conf)

    for i ,batch in enumerate(train_dl):
        model.module.forward(batch, 0.0)
        model.module.forward(batch, 0.1)
        model.module.forward(batch, 0.2)
        model.module.forward(batch, 0.3)
        model.module.forward(batch, 0.4)
        model.module.forward(batch, 0.5)
        model.module.forward(batch, 0.6)
        model.module.forward(batch, 0.7)
        model.module.forward(batch, 0.8)
        model.module.forward(batch, 0.9)
        model.module.forward(batch, 1.0)
        exit()
        # Logging on terminal
        sys.stdout.write('\n[%04d/%04d] %d'% (i+1, num_batches, i))
        sys.stdout.flush()
