import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, input_nc, use_dropout):
        super(ResBlock, self).__init__()

        res_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, input_nc, 3),
            nn.InstanceNorm2d(input_nc),
            nn.ReLU(inplace=True)  # inplace : input으로 들어온 tensor자체를 변경 --> memory 절약
        ]

        if use_dropout:
            res_block += [nn.Dropout(0.5)]

        res_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, input_nc, 3),
            nn.InstanceNorm2d(input_nc)
        ]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, inputs):
        # residual block 이므로 output에 x를 더해서 return
        return self.res_block(inputs) + inputs


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, use_dropout=False, n_res_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_res_blocks):
            model += [ResBlock(in_features, use_dropout)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(512, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return torch.squeeze(x, 0)

