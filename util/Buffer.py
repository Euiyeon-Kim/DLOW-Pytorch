import torch
import random
import numpy as np


# CycleGAN에서 학습 안정화를 위해 사용 --> DLOW 모델에서는 사용하지 않음

class ImageBuffer():
    def __init__(self, buf_size):
        '''
            이전에 generate된 이미지를 저장하기 위한 버퍼
            --> Discriminator를 훈련시킬 때 이전에 생성된 이미지를 활용할 수 있다
        '''
        self.buf_size = buf_size
        if buf_size>0:
            self.num_imgs = 0
            self.imgs = []

    def query(self, imgs):
        '''
            Buffer에서 이미지를 찾아서 return
            imgs : latest generator가 생성한 이미지들
            50%의 확률로 input image를 return
            50%의 확률로 이전에 저장된 image를 return하고 현재 이미지를 buffer에 저장
        '''
        if self.buf_size==0: # 미리 저장된 이미지가 없다면 input을 그대로 반환
            return imgs

        response = []
        for img in imgs:
            img = torch.unsqueeze(img.data, 0) # (input, dim) --> dim이 insert된 tensor를 return
            if self.num_imgs < self.buf_size: # Buffer에 있는 data의 수가 buffer 크기보다 적음
                self.num_imgs += 1
                self.imgs.append(img)
                response.append(img)
            else:
                if random.uniform(0, 1) > 0.5:
                    replace = random.randint(0, self.buf_size-1) # Buffer에서 제거할 image선택
                    tmp = self.imgs[replace].clone()
                    self.imgs[replace] = img
                    response.append(tmp)
                else:
                    response.append(img)
        response = torch.cat(response)
        return response
