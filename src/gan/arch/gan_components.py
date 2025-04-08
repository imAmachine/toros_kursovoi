import torch
import torch.nn as nn
from torch.nn import Sequential

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True, activation=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_channels=2, feature_maps=64, extra_upsample=True):
        super(Generator, self).__init__()

        # Энкодер
        self.enc1 = ConvBlock(input_channels, feature_maps, use_bn=False, activation=nn.LeakyReLU(0.2))
        self.enc2 = ConvBlock(feature_maps, feature_maps*2, activation=nn.LeakyReLU(0.2))
        self.enc3 = ConvBlock(feature_maps*2, feature_maps*4, activation=nn.LeakyReLU(0.2))
        self.enc4 = ConvBlock(feature_maps*4, feature_maps*8, activation=nn.LeakyReLU(0.2))


        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        if not extra_upsample:
            self.model = Sequential(
                *downsample(input_channels, feature_maps, normalize=False),
                *downsample(feature_maps, feature_maps),
                *downsample(feature_maps, feature_maps*2),
                *downsample(feature_maps*2, feature_maps*4),
                *downsample(feature_maps*4, feature_maps*8),
                nn.Conv2d(64, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=1),
                nn.ReLU()
                *upsample(feature_maps*8, feature_maps*4),
                *upsample(feature_maps*4, feature_maps*2),
                *upsample(feature_maps*2, feature_maps),
                nn.Conv2d(feature_maps, 1, 3, 1, 1),
                nn.Tanh()
            )
        else:
            self.model = Sequential(
                *downsample(input_channels, feature_maps, normalize=False),
                *downsample(feature_maps, feature_maps),
                *downsample(feature_maps, feature_maps*2),
                *downsample(feature_maps*2, feature_maps*4),
                *downsample(feature_maps*4, feature_maps*8),
                nn.Conv2d(feature_maps*8, 4000, 1),
                *upsample(4000, feature_maps*8),
                *upsample(feature_maps*8, feature_maps*4),
                *upsample(feature_maps*4, feature_maps*2),
                *upsample(feature_maps*2, feature_maps),
                *upsample(feature_maps, feature_maps),
                nn.Conv2d(feature_maps, 1, 3, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, x, mask):
        x_combined = torch.cat([x, mask], dim=1)
        output = self.model(x_combined)
        composite = x * mask + output * (1 - mask)
        return composite, output

class Discriminator(nn.Module):
    def __init__(self, input_channels=1, feature_maps=16):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = input_channels
        for out_filters, stride, normalize in [(feature_maps, 2, False), (feature_maps*2, 2, True), (feature_maps*4, 2, True), (feature_maps*8, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = Sequential(*layers)

    def forward(self, img):
        return self.model(img)