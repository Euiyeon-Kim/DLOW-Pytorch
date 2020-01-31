'''
    Objective : Seperate dataset into train, validation, test sets
                Image preprocessing (비율 유지, 가로 1024로 resizing)
    
    * Before
    └─ Cityscapes (Cs_dir)
	   ├─ train
	   │  ├─ img
       │  │  └─ (folders)
	   │  └─ label
       │     └─ (folders)
       ├─ val
	   │  ├─ img
       │  │  └─ (folders)
	   │  └─ label
       │     └─ (folders)
	   └─ test
	      ├─ img
          │  └─ (folders)
	      └─ label
             └─ (folders)

    * After
    └─ Target (T_dir)
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
parser.add_argument('--T_dir', type=str, default='../datasets/Target', help="Where to save resized datasets")
parser.add_argument('--Cs_dir', type=str, default='../datasets/Cityscapes', help="Where to find Cityscapes datasets")
parser.add_argument('--img_format', type=str, default='.png', help="Image data type")


def resize_and_save(filepath, outputpath):
    image = Image.open(filepath)
    w, h = image.size
    new_h = int(1024 / w * h)
    image = image.resize((1024, new_h), Image.NEAREST)    # Segmentation task이므로 nearest neighbor interpolation
    image.save(outputpath)


if __name__  == '__main__':

    args = parser.parse_args()

    # assert 뒤에 오는 구문이 사실이 아니라면 error
    assert os.path.isdir(args.Cs_dir), "Dataset directory doesn't exist"
    if not os.path.isdir(args.T_dir):
        os.mkdir(args.T_dir)

    for split in ['train', 'val', 'test']:
        # Check cityscapes dataset exists
        split_dir = os.path.join(args.Cs_dir, split)
        assert os.path.isdir(split_dir), "Cityscapes "+split+" directory doesn't exists"
        split_imgdir = os.path.join(split_dir, 'img')
        split_labeldir = os.path.join(split_dir, 'label')
        assert os.path.isdir(split_imgdir), "Cityscapes "+split+" image directory doesn't exists"
        assert os.path.isdir(split_labeldir), "Cityscapes "+split+" label directory doesn't exists"

        # Make output directories
        output_dir = os.path.join(args.T_dir, split)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_imgdir = os.path.join(output_dir, 'img')
        output_labeldir = os.path.join(output_dir, 'label')
        if not os.path.exists(output_imgdir):
            os.mkdir(output_imgdir)
        if not os.path.exists(output_labeldir):
            os.mkdir(output_labeldir)

        img_names = []
        folders = os.listdir(split_imgdir)
        for folder in folders:
            if folder[0]!='.':
                imgdir = os.path.join(split_imgdir, folder)
                labeldir = os.path.join(split_labeldir, folder)
                if os.path.isdir(imgdir):
                    img_names = os.listdir(imgdir)

                print("Processing {}-{} data, saving preprocessed data".format(split, folder))
                for img_name in tqdm(img_names):
                    tmp = os.path.splitext(img_name)
                    label_name = tmp[0][:-12]+"_gtFine_color"+tmp[1]
                    output_imgname = tmp[0][:-12]+tmp[1]
                    resize_and_save(os.path.join(imgdir, img_name), os.path.join(output_imgdir, output_imgname))
                    resize_and_save(os.path.join(labeldir, label_name), os.path.join(output_labeldir, output_imgname))
