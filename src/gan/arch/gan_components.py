import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .common import BaseNetwork
import torch.nn as nn

class AOTGenerator(BaseNetwork):
    def __init__(self, rates, block_num, input_channels=2, feature_maps=64):
        super(AOTGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, feature_maps, 7),
            nn.ReLU(True),
            nn.Conv2d(feature_maps, feature_maps*2, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature_maps*2, feature_maps*4, 4, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(*[AOTBlock(feature_maps*4, rates) for _ in range(block_num)])

        self.decoder = nn.Sequential(
            UpConv(feature_maps*4, feature_maps*2), 
            nn.ReLU(True), 
            UpConv(feature_maps*2, feature_maps), 
            nn.ReLU(True), 
            nn.Conv2d(feature_maps, 1, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

import torch.nn as nn
class AOTDiscriminator(BaseNetwork):
    def __init__(self, input_channels=1, feature_maps=64):
        super(AOTDiscriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, feature_maps, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_maps, feature_maps*2, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_maps*2, feature_maps*4, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_maps*4, feature_maps*8, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps*8, 1, 4, stride=1, padding=1),
        )

        self.init_weights()
    
    def forward(self, x):
        return self.model(x)
