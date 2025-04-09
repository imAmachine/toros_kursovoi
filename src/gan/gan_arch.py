import torch
import torch.nn as nn


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


class GanGenerator(nn.Module):
    def __init__(self, input_channels=2, feature_maps=64):
        super(GanGenerator, self).__init__()
        
        # --- Энкодер ---
        self.enc1 = ConvBlock(input_channels, feature_maps, use_bn=False, activation=nn.LeakyReLU(0.2))
        self.enc2 = ConvBlock(feature_maps, feature_maps * 2, activation=nn.LeakyReLU(0.2))
        self.enc3 = ConvBlock(feature_maps * 2, feature_maps * 4, activation=nn.LeakyReLU(0.2))
        self.enc4 = ConvBlock(feature_maps * 4, feature_maps * 8, activation=nn.LeakyReLU(0.2))
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_maps * 8, feature_maps * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_maps * 8, feature_maps * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_maps * 8, feature_maps * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
        )
        
        # --- Декодер с учетом concat ---
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 4, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 2, feature_maps // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(feature_maps // 2, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x, mask):
        x_combined = torch.cat([x, mask], dim=1)
        
        # --- Кодирование ---
        e1 = self.enc1(x_combined)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # --- Bottleneck ---
        bn = self.bottleneck(e4)
        
        # --- Декодирование с пропусками ---
        d1 = self.dec1(bn)
        d1 = torch.cat([d1, e3], dim=1)
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        
        d4 = self.dec4(d3)
        
        output = self.final(d4)
        return output


class GanDiscriminator(nn.Module):
    def __init__(self, input_channels=1, feature_maps=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps*2, feature_maps*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps*4, feature_maps*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps*8, 1, 4, 1, 0, bias=False)
        )
    
    def forward(self, x):
        return self.layers(x)
