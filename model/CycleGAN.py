import numpy as np
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, input_nc, use_dropout):
        super(ResBlock, self).__init__()

        res_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, input_nc, 3),
            nn.InstanceNorm2d(input_nc),
            nn.ReLU(inplace=True)
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
    def __init__(self, input_nc, output_nc, output_filter_num=64, use_dropout=False, n_residual_blocks=9):
        super(Generator, self).__init__()
        '''
            output_filter_num : number of filters in the last conv layer
            n_residual_blocks : 9 for 256x256 images / 6 for 128x128 images at CycleGAN
        '''
        # Initial convolution block     
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, output_filter_num, kernel_size=7, padding=0),
            nn.InstanceNorm2d(output_filter_num),
            nn.ReLU(inplace=True) # inplace : Change input itself --> Can save memory
        ]
              
        n_downsampling = 2
        for i in range(n_downsampling): # Downsampling
            mult = 2 ** i
            model += [
                nn.Conv2d(output_filter_num * mult, output_filter_num * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(output_filter_num * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_residual_blocks): # Resnet blocks
            model += [ResBlock(output_filter_num * mult, use_dropout=use_dropout)]

        for i in range(n_downsampling):  # Upsampling
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(output_filter_num * mult, int(output_filter_num * mult / 2), kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(int(output_filter_num * mult / 2)),
                nn.ReLU(inplace=True)
            ]
            
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(output_filter_num, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, inputs):
        return self.model(inputs)


class Discriminator(nn.Module): # PatchGAN
    def __init__(self, input_nc, output_filter_num=64, n_layers=3):
        """
            output_filter_num : number of filters in the last conv layer
        """
        super(Discriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, output_filter_num, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        # nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(output_filter_num * nf_mult_prev, output_filter_num * nf_mult, kernel_size=kw, stride=2),
                nn.BatchNorm2d(output_filter_num * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(output_filter_num * nf_mult_prev, output_filter_num * nf_mult, kernel_size=kw, stride=1),
            nn.BatchNorm2d(output_filter_num * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(output_filter_num * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, inputs):
        return self.model(inputs)