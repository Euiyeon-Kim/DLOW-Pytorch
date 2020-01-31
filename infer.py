import os
import sys
import yaml
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm # for visualized logging

from torchvision.transforms import transforms

from util.Logger import Logger
from model.InterpolationGAN import InterpolationGAN

# 이미지 크기에 따라 달라짐 --> 400, 400으로 crop할 영역
areas = [
            (0, 0), (0, 400), (0, 800), (0, 1200), (0, 1914-400),
            (400, 0), (400, 400), (400, 800), (400, 1200), (400, 1914-400), 
            (1052-400, 0), (1052-400, 400), (1052-400, 800), (1052-400, 1200), (1052-400, 1914-400), 
        ]
infer_T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


if __name__ == "__main__":
    # Reading configuration file
    conf = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    infer_conf = conf['infer']

    # Checkpoint directory
    if not os.path.exists(os.path.join(infer_conf['checkpoint_dir'], "last.pth.tar")):
        print(os.path.join(infer_conf['checkpoint_dir'], "last.pth.tar"))
        print("Checkpoint file doesn't exists")
        exit()

    # Directory 생성
    if not os.path.isdir(infer_conf['DLOW_dir']):
        os.mkdir(infer_conf['DLOW_dir'])
    img_path = os.path.join(infer_conf['DLOW_dir'], 'img')
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    label_path = os.path.join(infer_conf['DLOW_dir'], 'label')
    if not os.path.isdir(label_path):
        os.mkdir(label_path)

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
    #print(model)
    model = torch.nn.DataParallel(model, output_device=1)
    model.to(device)
    model.module.load(os.path.join(infer_conf['checkpoint_dir'], "last.pth.tar"))

    for param in model.parameters():
        param.requires_grad = False
    
    # Inference에 사용할 이미지 목록
    infer_img_path = os.path.join(infer_conf['infer_dir'], 'img')
    infer_label_path = os.path.join(infer_conf['infer_dir'], 'label')

    filenames = sorted(os.listdir(infer_img_path))
    filenames = [os.path.join(infer_img_path, f) for f in filenames if f.endswith('.png')]
    length = len(filenames)
    
    for i in range(length):
        filename = filenames[i]
        label = Image.open(os.path.join(infer_label_path, filename.split('/')[-1]))

        z = round(random.random(), 2)

        img = np.array(Image.open(filename))
        result = np.zeros_like(img)

        for area in areas:
            w_start, h_start = area
            crop = torch.unsqueeze(infer_T(Image.fromarray(img[w_start:w_start+400, h_start:h_start+400])), 0)
            tmp_result = np.array(model.module.forward(crop, z))
            result[w_start:w_start+400, h_start:h_start+400] = tmp_result.transpose(1, 2, 0)

        result = Image.fromarray(result)
        result.save(os.path.join(img_path, str(z)+"_"+filename.split("/")[-1]))
        
        label.save(os.path.join(label_path, str(z)+"_"+filename.split("/")[-1]))

        sys.stdout.write('\n[%04d/%04d]'% (i+1, length))
        sys.stdout.flush()

    # Make 
    filenames = sorted(os.listdir(img_path))
    filenames = [ filename for filename in filenames if filename.endswith('.png')]
    DLOW_list = open(os.path.join(infer_conf['DLOW_list_dir'], "Atrain.txt"), 'w')
    for filename in tqdm(filenames):
        name = filename+"\n"
        DLOW_list.write(name)
    DLOW_list.close()