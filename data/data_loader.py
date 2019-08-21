import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

'''
Source Dataset : GTA5 (labeled)
Target Dataset : bdd100k (unlabeled)
Goal : Convert Source data to Target data distribution
'''

W = 1024
H = 576

S_transformer = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

T_transformer = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


# Transformer for train data
class Image_Dataset(Dataset):
    def __init__(self, S_dir, T_dir, S_transform, T_transform):
        S_image_path = os.path.join(S_dir, 'img')
        #S_label_path = os.path.join(S_dir, 'label')
        T_image_path = os.path.join(T_dir, 'img')
        #T_label_path = os.path.join(T_dir, 'label')

        self.S_filenames = sorted(os.listdir(S_image_path))
        self.T_filenames = sorted(os.listdir(T_image_path))
        self.S_filenames = [os.path.join(S_image_path, f) for f in self.S_filenames if f.endswith('.png')]
        self.T_filenames = [os.path.join(T_image_path, f) for f in self.T_filenames if f.endswith('.jpg')]
         
        #self.S_labelnames = sorted(os.listdir(S_label_path)) # Source dataset의 Label for training
        #self.T_labelnames = sorted(os.listdir(T_label_path)) # Target dataset의 Label for evaluation
        #self.S_labelnames = [os.path.join(S_label_path, l) for l in self.S_labelnames if l.endswith('.jpg')]
        #self.T_labelnames = [os.path.join(T_label_path, l) for l in self.T_labelnames if l.endswith('.jpg')]

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

        return {'S_img': S_img, 'T_img': T_img} #, 'S_label': S_label, 'T_label': T_label


def get_dataloaders(types, S_dir, T_dir, params):
    dataloaders = {}

    for option in ['train', 'val', 'test']:
        if option in types:
            S_path = os.path.join(S_dir, option)
            T_path = os.path.join(T_dir, option)
            data_loader = DataLoader(Image_Dataset(S_path, T_path, S_transformer, T_transformer),
                                     batch_size=params.batch_size, shuffle=True,
                                     num_workers=params.num_workers, pin_memory=params.cuda)
            dataloaders[option] = data_loader

    return dataloaders

