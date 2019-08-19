import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

'''
Source Dataset : GTA5 (labeled)
Target Dataset : bdd100k (unlabeled)
Goal : Convert Source data to Target data distribution
'''

# Transformer for train data
class Image_Dataset(Dataset):
    def __init__(self, S_dir, T_dir, S_transform, T_transform):
        self.S_filenames = os.listdir(S_dir)
        self.T_filenames = os.listdir(T_dir)
        self.S_filenames = [os.path.join(S_dir, f) for f in self.S_filenames if f.endswith('.jpg')]
        self.T_filenames = [os.path.join(T_dir, f) for f in self.T_filenames if f.endswith('.jpg')]
         
        # self.S_label = Source dataset의 Label for training
        # self.T_label = Target dataset의 Label for evaluation

        self.S_size = len(self.S_filenames)
        self.T_size = len(self.T_filenames)

        self.S_transform = S_transform
        self.T_transform = T_transform

    def __len__(self):
        return max(self.S_size, self.T_size)

    def __getitem__(self, idx):
        S_img = Image.open(self.S_filenames[idx%self.S_size])
        T_img = Image.open(self.T_filenames[idx%self.T_size])
        S_img = self.S_transform(S_img)
        T_img = self.T_transform(T_img)

        # S_label = self.S_label[idx%self.S_size]
        # T_label = self.T_label[idx%self.T_size]

        return S_img, T_img #, S_label, T_label


'''def fetch_dataloader(types, S_dir, T_dir, params):
    dataloaders = {}

    for option in ['train', 'val', 'test']:
        if option in types:
            S_path = path.os.join(S_dir, option)
            T_path = path.os.join(T_dir, option)

            if option =='train':'''
