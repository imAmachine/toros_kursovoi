import torch
import torch.nn.functional as F

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
    
class AOTGenerator(torch.nn.Module):
    def __init__(self, input_channels=2, feature_maps=64):
        super().__init__()
        # Encoder
        self.enc1 = GatedConv(input_channels, feature_maps, kernel_size=7, padding=3)
        self.enc2 = GatedConv(feature_maps, feature_maps*2, kernel_size=4, stride=2, padding=1)
        self.enc3 = GatedConv(feature_maps*2, feature_maps*4, kernel_size=4, stride=2, padding=1)
        
        # AOT Blocks - несколько блоков вместо одного для лучшего улавливания паттернов
        self.aot_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(feature_maps*4, feature_maps*4, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(feature_maps*4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(feature_maps*4, feature_maps*4, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(feature_maps*4)
            ) for _ in range(3)  # Добавляем 3 блока для более глубоких признаков
        ])
        
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
        
        # Attention механизм
        att = self.attention(x)
        x = x * att
        
        # Decode с skip-соединениями для сохранения деталей
        x = self.dec1(x)
        x = x + e2  # Skip connection
        x = self.dec2(x)
        x = x + e1  # Skip connection
        x = self.dec3(x, mask)

        generated = self.sigmoid(x)
        
        return self.sigmoid(x)


class AOTDiscriminator(torch.nn.Module):
    def __init__(self, input_channels=1, feature_maps=64):
        super(AOTDiscriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            torch.nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(feature_maps * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(feature_maps * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(feature_maps * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0)
        )
    
    def forward(self, x):
        return self.model(x)
