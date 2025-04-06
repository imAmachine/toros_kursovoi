import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class GatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels*2, kernel_size, stride, padding)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, mask):
        out = self.conv(x)
        features, gate = torch.split(out, out.shape[1]//2, dim=1)
        if mask.shape[2:] != gate.shape[2:]:
            mask = F.interpolate(mask, size=gate.shape[2:], mode='bilinear', align_corners=False)
        gate = self.sigmoid(gate + mask)
        return features * gate
    
class AOTBlock(torch.nn.Module):
    def __init__(self, dim, rates=(1, 2, 4, 8)):
        super(AOTBlock, self).__init__()
        self.rates = rates
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.ReflectionPad2d(rate),
                torch.nn.Conv2d(dim, dim // 4, kernel_size=3, padding=0, dilation=rate),
                torch.nn.ReLU(True),
            )
            for rate in rates
        ])
        self.fuse = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=0),
        )
        self.gate = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=0),
        )

    def forward(self, x):
        out = torch.cat([blk(x) for blk in self.blocks], dim=1)
        out = self.fuse(out)
        gate = torch.sigmoid(self._layer_norm(self.gate(x)))
        return x * (1 - gate) + out * gate

    def _layer_norm(self, feat):
        mean = feat.mean((2, 3), keepdim=True)
        std = feat.std((2, 3), keepdim=True) + 1e-9
        return 5 * (2 * (feat - mean) / std - 1)
    
class AOTGenerator(torch.nn.Module):
    def __init__(self, input_channels=2, feature_maps=64, aot_blocks=2):
        super().__init__()
        # Encoder
        self.enc1 = GatedConv(input_channels, feature_maps, kernel_size=7, padding=3)
        self.enc2 = GatedConv(feature_maps, feature_maps*2, kernel_size=4, stride=2, padding=1)
        self.enc3 = GatedConv(feature_maps*2, feature_maps*4, kernel_size=4, stride=2, padding=1)
        
        # AOT Blocks - несколько блоков вместо одного для лучшего улавливания паттернов
        self.aot_blocks = torch.nn.ModuleList([AOTBlock(dim=feature_maps*4) for _ in range(aot_blocks)])
        
        # Attention module - для лучшего восстановления структурных деталей
        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(feature_maps*4, feature_maps, kernel_size=1),
            torch.nn.BatchNorm2d(feature_maps),
            torch.nn.ReLU(),
            torch.nn.Conv2d(feature_maps, feature_maps*4, kernel_size=1),
            torch.nn.BatchNorm2d(feature_maps*4),
            torch.nn.Sigmoid()
        )
        
        # Decoder
        self.dec1 = torch.nn.ConvTranspose2d(feature_maps*4, feature_maps*2, kernel_size=4, stride=2, padding=1)
        self.dec2 = torch.nn.ConvTranspose2d(feature_maps*2, feature_maps, kernel_size=4, stride=2, padding=1)
        self.dec3 = GatedConv(feature_maps, 1, kernel_size=7, padding=3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, mask):
        # Объединяем входное изображение и маску
        x_input = x
        x = torch.cat([x, mask], dim=1)
        
        # Encode
        e1 = self.enc1(x, mask)
        e2 = self.enc2(e1, mask)
        e3 = self.enc3(e2, mask)
        
        # AOT Blocks с остаточными соединениями
        x = e3
        for aot_block in self.aot_blocks:
            x = x + aot_block(x)
        
        # Decode с skip-соединениями для сохранения деталей
        x = self.dec1(x)
        x = x + e2  # Skip connection
        x = self.dec2(x)
        x = x + e1  # Skip connection
        x = self.dec3(x, mask)
        
        return self.sigmoid(x)


class AOTDiscriminator(torch.nn.Module):
    def __init__(self, input_channels=1, feature_maps=64):
        super(AOTDiscriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(torch.nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(torch.nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(torch.nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0)
        )
    
    def forward(self, x):
        return self.model(x)
