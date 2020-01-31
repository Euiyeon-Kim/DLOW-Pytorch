'''
    Objective : Image preprocessing (비율 유지, 가로 1024로 resizing)
    
    * Before
    └─ GTA (GTA_dir)
	   ├─ img
       └─ label
    
    * After
    └─ Source (S_dir)
	   ├─ img
       └─ label
'''

import os
import shutil
import random
import argparse

from PIL import Image
from tqdm import tqdm # for visualized logging


parser = argparse.ArgumentParser()
parser.add_argument('--GTA_dir', type=str, default='../datasets/GTA5', help="Where to find GTA5 datasets")
parser.add_argument('--S_dir', type=str, default='../datasets/Source', help="Where to save resized datasets")
parser.add_argument('--S_list_dir', type=str, default="../datasets/Source_list", help="Where to save GTA5 datasets file list")
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
    if not os.path.isdir(args.S_list_dir):
        os.mkdir(args.S_list_dir)

    filenames = sorted(os.listdir(imgdir))
    filenames = [ filename for filename in filenames if filename.endswith(args.img_format)]

    output_imgdir = os.path.join(args.S_dir, 'img')
    output_labeldir = os.path.join(args.S_dir, 'label')
    if not os.path.exists(output_imgdir):
        os.mkdir(output_imgdir)
    if not os.path.exists(output_labeldir):
        os.mkdir(output_labeldir)
    
    print("Processing data, saving preprocessed data")
    S_list = open(os.path.join(args.S_list_dir, "Btrain.txt"), 'w')
    for filename in tqdm(filenames):
        name = filename+"\n"
        S_list.write(name)
        resize_and_save(os.path.join(imgdir, filename), os.path.join(output_imgdir, filename))
        resize_and_save(os.path.join(labeldir, filename), os.path.join(output_labeldir, filename))
    S_list.close()
