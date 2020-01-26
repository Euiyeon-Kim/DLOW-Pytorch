'''
    Objective : Seperate dataset into train, validation, test sets
                Image preprocessing (비율 유지, 가로 1024로 resizing)
    
    * Before
    └─ GTA (GTA_dir)
	   ├─ img
       └─ label
    
    * After
    └─ Source (S_dir)
	   ├─ train
	   │  ├─ img
	   │  └─ label
       ├─ val
	   │  ├─ img
	   │  └─ label
	   └─ test
	      ├─ img
	      └─ label
'''

import argparse
import shutil
import random
import os

from PIL import Image
from tqdm import tqdm # for visualized logging


parser = argparse.ArgumentParser()
parser.add_argument('--S_dir', type=str, default='../datasets/Source', help="Where to save splited and resized datasets")
parser.add_argument('--GTA_dir', type=str, default='../datasets/GTA5', help="Where to find GTA5 datasets")
parser.add_argument('--train_ratio', type=float, default=0.7, help="Train datas' ratio")
parser.add_argument('--val_ratio', type=float, default=0.15, help="Validation datas' ratio")
parser.add_argument('--img_format', type=str, default='.png', help="Image data type")


def resize_and_save(filepath, outputpath):
    image = Image.open(filepath)
    w, h = image.size
    new_h = int(1024 / w * h)
    image = image.resize((1024, new_h), Image.NEAREST)    # Segmentation task이므로 nearest neighbor interpolation
    image.save(outputpath)


if __name__  == '__main__':

    args = parser.parse_args()
    imgdir = os.path.join(args.GTA_dir, 'img')
    labeldir = os.path.join(args.GTA_dir, 'label')

    # assert 뒤에 오는 구문이 사실이 아니라면 error
    assert os.path.isdir(imgdir), "Dataset image directory doesn't exist"
    assert os.path.isdir(labeldir), "Dataset label directory doesn't exist"
    if not os.path.isdir(args.S_dir):
        os.mkdir(args.S_dir)

    filenames = os.listdir(imgdir)
    filenames = [ filename for filename in filenames if filename.endswith(args.img_format)]

    # Split validation set and train set
    random.seed(2020)
    filenames.sort()
    random.shuffle(filenames)

    train_split = int(args.train_ratio*len(filenames))
    val_split = train_split+int(args.val_ratio*len(filenames))
    train_filenames = filenames[:train_split]
    val_filenames = filenames[train_split:val_split]
    test_filenames = filenames[val_split:]
    filenames = {'train':train_filenames, 'val':val_filenames, 'test':test_filenames}

    for split in ['train', 'val', 'test']:
        output_dir = os.path.join(args.S_dir, split)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_imgdir = os.path.join(output_dir, 'img')
        output_labeldir = os.path.join(output_dir, 'label')
        if not os.path.exists(output_imgdir):
            os.mkdir(output_imgdir)
        if not os.path.exists(output_labeldir):
            os.mkdir(output_labeldir)
       
        print("Processing {} data, saving preprocessed data".format(split))
        for filename in tqdm(filenames[split]):
            resize_and_save(os.path.join(imgdir, filename), os.path.join(output_imgdir, filename))
            resize_and_save(os.path.join(labeldir, filename), os.path.join(output_labeldir, filename))

