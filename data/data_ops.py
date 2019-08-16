from torchvision.transforms import transforms
from PIL import Image

H = 576 # 720
W = 1024 # 1280

train_transformer = transforms.Compose([
    transforms.Resize((H, W), Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transformer = transforms.Compose([
    transforms.Resize((H, W), Image.BICUBIC),
    transforms.ToTensor()
])


def get_transform(params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append()

    

if __name__ =='__main__':
    transform_path = './transform.json'
    get_transform(transform_path)