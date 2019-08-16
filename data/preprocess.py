import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--Source_dir', default='./GTA5', help="Where to find Source dataset")
parser.add_argument('--Target_dir', default='./bdd100k', help="Where to find Target dataset")
parser.add_argument('--Source_output_dir', default='./GTA5', help="Where to write Source dataset")
parser.add_argument('--Target_output_dir', default='./bdd100k', help="Where to write Target dataset")

def resize_and_save(filepath, output_dir, size=SIZE):
    image = Image.open(filepath)
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filepath.split('/')[-1]))


if __name__  == '__main__':
    args = parser.parse_args()

    # assert 뒤에 오는 구문이 사실이 아니라면 error
    assert os.path.isdir(args.data_dir), "Dataset directory doesn't exist"

    train_data_dir = os.path.join(args.data_dir, 'train_signs')
    test_data_dir = os.path.join(args.data_dir, 'test_signs')

    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, filename) for filename in filenames if filename.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, filename) for filename in test_filenames if filename.endswith('.jpg')]

    # Split validation set and train set
    random.seed(2019)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8*len(filenames)) # 80% of train data is training dataset
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {'train':train_filenames, 'val':val_filenames, 'test':test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Output dir {} already exists".format(args.output_dir))

    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_signs'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

