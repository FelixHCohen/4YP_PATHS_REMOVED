import torch
import torch.nn as nn


def norm(input: torch.tensor, norm_name: str):
    if norm_name == 'layer':
        normaliza = nn.LayerNorm(list(input.shape)[1:])
    elif norm_name == 'batch':
        normaliza = nn.BatchNorm2d(list(input.shape)[1])
    elif norm_name == 'instance':
        normaliza = nn.InstanceNorm2d(list(input.shape)[1])

    normaliza = normaliza.to(f'cuda:{input.get_device()}')

    output = normaliza(input)

    return output
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, norm_name='batch'):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(out_c, affine=False)
        self.norm2 = nn.BatchNorm2d(out_c, affine=False) #don't need to define 2 norms as we have set affine to false, here if one decides to change to affine=true
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)

        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, norm_name):
        super().__init__()

        self.conv = conv_block(in_c, out_c, norm_name)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, norm_name):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, norm_name)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_c, out_c, base_c,kernels = [2,4,8,16], norm_name = 'batch'):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(in_c, base_c, norm_name)
        self.e2 = encoder_block(base_c, base_c*kernels[0], norm_name)
        self.e3 = encoder_block(base_c*kernels[0], base_c*kernels[1], norm_name)
        self.e4 = encoder_block(base_c*kernels[1], base_c*kernels[2], norm_name)

        """ Bottleneck """
        self.b = conv_block(base_c*kernels[2], base_c*kernels[3], norm_name)

        """ Decoder """
        self.d1 = decoder_block(base_c*kernels[3], base_c*kernels[2], norm_name)
        self.d2 = decoder_block(base_c*kernels[2], base_c*kernels[1], norm_name)
        self.d3 = decoder_block(base_c*kernels[1], base_c*kernels[0], norm_name)
        self.d4 = decoder_block(base_c*kernels[0], base_c, norm_name)

        """ Classifier """
        self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs