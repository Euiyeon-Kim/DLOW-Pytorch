import torch


class ImageBuffer():

    def __init__(self, buf_size):
        self.buf_size = buf_size
        if buf_size>0:
            self.num_imgs = 0
            self.images = []

    def query(self, imgs):
        pass
