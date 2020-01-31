import os
import sys
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

'''
    Source Dataset : GTA5 (labeled)
    Target Dataset : bdd100k (unlabeled)
'''

class DLOWDataset(Dataset):
    def __init__(self, root_dir, S_transform, T_transform, fixed_pair=False, mode='train'):
        '''
            root_dir : Dataset directory
            fixed_pair : Source dataset과 Target dataset의 pair가 fixed하지 않도록 randomize
            혹시 모를 경우를 대비해서 확장자 확인
        '''
        S_image_path = os.path.join(root_dir, 'Source', 'img')
        T_image_path = os.path.join(root_dir, 'Target', mode, 'img')

        self.S_imgnames = sorted(os.listdir(S_image_path))
        self.T_imgnames = sorted(os.listdir(T_image_path))
        self.S_imgnames = [os.path.join(S_image_path, f) for f in self.S_imgnames if f.endswith('.png')]
        self.T_imgnames = [os.path.join(T_image_path, f) for f in self.T_imgnames if f.endswith('.png')]

        self.fixed_pair = fixed_pair
        self.S_size = len(self.S_imgnames)
        self.T_size = len(self.T_imgnames)
        self.S_transform = S_transform
        self.T_transform = T_transform

    def __len__(self): # 두 데이터 셋의 크기가 다를 수 있으므로 max
        return max(self.S_size, self.T_size)

    def __getitem__(self, idx):
        S_name = self.S_imgnames[idx%self.S_size]
        S_img = self.S_transform(Image.open(S_name))

        if self.fixed_pair:
            T_name = self.T_imgnames[idx%self.T_size]
            T_img = self.T_transform(Image.open(T_name))
        else:
            T_name = self.T_imgnames[random.randint(0, self.T_size-1)]
            T_img = self.T_transform(Image.open(T_name))

        return {'S_name': S_name, 'S_img': S_img, 'T_name': T_name, 'T_img': T_img}


def get_transformer(S_H, T_H, is_train=True):
    '''
        preprocessing 코드에서 resizing 후 새로 dataset을 저장하므로 transforms.Resize를 모두 주석
        preprocessing 없이 바로 train.py 실행 시 resize 주석 해재 필요
        transforms.ToTensor() --> Tensor의 범위가 [0, 1]
        transfroms.Normarlize((0.5, ), (0.5, ))는 [0, 1] --> [-1, 1]로 변환
    '''
    if is_train:                               # transformer for train.py 
        S_transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize((S_H, 1024)),
            transforms.RandomCrop((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # (mean), (std) --> -1 에서 1사이로 normalize
        ])

        T_transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize((T_H, 1024)),
            transforms.RandomCrop((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return S_transformer, T_transformer


def get_DLOW_dataloaders(types, conf, is_train=True):
    dataloaders = {}

    S_transformer, T_transformer = get_transformer(conf['S_H'], conf['T_H'], is_train)
    for option in ['train', 'val', 'test']:
        if option in types:
            data_loader = DataLoader(DLOWDataset(conf['data_root_dir'], S_transformer, T_transformer,
                                                 conf['fixed_pair'], mode=option),
                                     batch_size=conf['batch_size'], shuffle=True,
                                     num_workers=conf['num_workers'], pin_memory=False)
                                     # 여기서 pin_memory를 True로 두면 data를 복사.
            dataloaders[option] = data_loader

    return dataloaders