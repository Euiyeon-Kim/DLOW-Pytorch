import os
import yaml
import torch

from data import DataLoader
from util.Logger import Logger
from model.InterpolationGAN import InterpolationGAN


if __name__ == "__main__":
    # Reading configuration file
    conf = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    train_conf = conf['train']
    infer_conf = conf['infer']

    # Checkpoint directory
    if not os.path.exists(os.path.join(infer_conf['checkpoint_dir'], "last.pth.tar")):
        print(os.path.join(infer_conf['checkpoint_dir'], "last.pth.tar"))
        print("Checkpoint file doesn't exists")
        exit()

    # GPU 할당
    os.environ["CUDA_VISIBLE_DEVICES"] = infer_conf['gpu_id']

    # Model parameter loading
    model = InterpolationGAN(train_conf, False)
    print(model)
    model = torch.nn.DataParallel(model, output_device=1)
    model.cuda()
    model.module.load(os.path.join(infer_conf['checkpoint_dir'], "last.pth.tar"))

    for param in model.parameters():
        param.requires_grad = False
    
    dataloaders = DataLoader.get_dataloaders(['train'], train_conf, False)
    train_dl = dataloaders['train']

    num_batches = len(train_dl)
    logger = Logger(train_conf)

    print(num_batches)
    for i ,batch in enumerate(train_dl):
        model.module.set_input(batch)
        model.module.foward()
        exit()