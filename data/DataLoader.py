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


class ImageDataset(Dataset):
    def __init__(self, root_dir, S_transform, T_transform, fixed_pair=False, mode='train'):
        '''
            root_dir : Dataset directory
            fixed_pair : Source dataset과 Target dataset의 pair가 fixed하지 않도록 randomize
            혹시 모를 경우를 대비해서 확장자 확인
        '''
        S_image_path = os.path.join(root_dir, mode, 'Source', 'img')
        S_label_path = os.path.join(root_dir, mode, 'Source', 'label')
        T_image_path = os.path.join(root_dir, mode, 'Target', 'img')
        T_label_path = os.path.join(root_dir, mode, 'Target', 'label')

        self.S_imgnames = sorted(os.listdir(S_image_path))
        self.T_imgnames = sorted(os.listdir(T_image_path))
        self.S_imgnames = [os.path.join(S_image_path, f) for f in self.S_imgnames if f.endswith('.png')]
        self.T_imgnames = [os.path.join(T_image_path, f) for f in self.T_imgnames if f.endswith('.png')]
        print(self.S_imgnames)
        # 상응하는 label path도 parsing 할 것 --> for segmentation

        self.fixed_pair = fixed_pair
        self.S_size = len(self.S_imgnames)
        self.T_size = len(self.T_imgnames)
        self.S_transform = S_transform
        self.T_transform = T_transform

    def __len__(self): # 두 데이터 셋의 크기가 다를 수 있으므로 max
        return max(self.S_size, self.T_size)

    def __getitem__(self, idx):
        S_img = self.S_transform(Image.open(self.S_imgnames[idx%self.S_size]))
        # S_label = self.S_label[idx%self.S_size]

        if self.fixed_pair:
            T_img = self.T_transform(Image.open(self.T_imgnames[idx%self.T_size]))
            # T_label = self.T_label[idx%self.T_size]
        else:
            T_img = self.T_transform(Image.open(self.T_imgnames[random.randint(0, self.T_size-1)]))
            # T_label = self.T_label[random.randint(0, self.T_size-1)]

        return {'S_img': S_img, 'T_img': T_img}  # 'S_label': S_label, 'T_label': T_label


def get_transformer(H, W):  # 현재는 resizing이 없느나, 추가 가능
    '''
        transforms.ToTensor() --> Tensor의 범위가 [0, 1]
        transfroms.Normarlize((0.5, ), (0.5, ))는 [0, 1] --> [-1, 1]로 변환
    '''
    S_transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # (mean), (std) --> -1 에서 1사이로 normalize
    ])

    T_transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return S_transformer, T_transformer


def get_dataloaders(types, conf):
    dataloaders = {}

    S_transformer, T_transformer = get_transformer(conf['resize_H'], conf['resize_W'])
    for option in ['train', 'val', 'test']:
        if option in types:
            data_loader = DataLoader(ImageDataset(conf['data_root_dir'], S_transformer, T_transformer,
                                                  conf['fixed_pair'], mode=option),
                                     batch_size=conf['batch_size'], shuffle=True,
                                     num_workers=conf['num_workers'], pin_memory=False)
                                     # 여기서 pin_memory를 True로 두면 data를 복사.
            dataloaders[option] = data_loader

    return dataloaders