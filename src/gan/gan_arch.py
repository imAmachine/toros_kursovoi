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
        
        # Энкодер
        self.enc1 = ConvBlock(input_channels, feature_maps, use_bn=False, activation=nn.LeakyReLU(0.2))
        self.enc2 = ConvBlock(feature_maps, feature_maps*2, activation=nn.LeakyReLU(0.2))
        self.enc3 = ConvBlock(feature_maps*2, feature_maps*4, activation=nn.LeakyReLU(0.2))
        self.enc4 = ConvBlock(feature_maps*4, feature_maps*8, activation=nn.LeakyReLU(0.2))
        
        # Декодер с пропускными соединениями
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps*8, feature_maps*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps*4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps*8, feature_maps*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps*4, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True)
        )
        
        # Выходной слой
        self.final = nn.Sequential(
            nn.ConvTranspose2d(feature_maps*2, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask):
        # Объединяем входное изображение и маску
        x_combined = torch.cat([x, mask], dim=1)
        
        # Кодирование
        e1 = self.enc1(x_combined)  # [batch, feature_maps, H/2, W/2]
        e2 = self.enc2(e1)          # [batch, feature_maps*2, H/4, W/4]
        e3 = self.enc3(e2)          # [batch, feature_maps*4, H/8, W/8]
        e4 = self.enc4(e3)          # [batch, feature_maps*8, H/16, W/16]
        
        # Декодирование с пропускными соединениями
        d1 = self.dec1(e4)          # [batch, feature_maps*4, H/8, W/8]
        d1 = torch.cat([d1, e3], dim=1)  # [batch, feature_maps*8, H/8, W/8]
        
        d2 = self.dec2(d1)          # [batch, feature_maps*2, H/4, W/4]
        d2 = torch.cat([d2, e2], dim=1)  # [batch, feature_maps*4, H/4, W/4]
        
        d3 = self.dec3(d2)          # [batch, feature_maps, H/2, W/2]
        d3 = torch.cat([d3, e1], dim=1)  # [batch, feature_maps*2, H/2, W/2]
        
        # Финальный выход
        output = self.final(d3)    # [batch, 1, H, W]
        
        # Комбинируем оригинальное изображение и сгенерированную часть
        composite = x *  mask + output * mask
        
        return composite, output


class GanDiscriminator(nn.Module):
    def __init__(self, input_channels=1, feature_maps=64):
        super(GanDiscriminator, self).__init__()
        
        # PatchGAN дискриминатор
        self.layer1 = ConvBlock(input_channels, feature_maps, 
                               kernel_size=4, stride=2, padding=1, 
                               use_bn=False, activation=nn.LeakyReLU(0.2))
        
        self.layer2 = ConvBlock(feature_maps, feature_maps*2, 
                               kernel_size=4, stride=2, padding=1, 
                               use_bn=True, activation=nn.LeakyReLU(0.2))
        
        self.layer3 = ConvBlock(feature_maps*2, feature_maps*4, 
                               kernel_size=4, stride=2, padding=1, 
                               use_bn=True, activation=nn.LeakyReLU(0.2))
        
        self.layer4 = ConvBlock(feature_maps*4, feature_maps*8, 
                               kernel_size=4, stride=1, padding=1, 
                               use_bn=True, activation=nn.LeakyReLU(0.2))
        
        # Финальный слой для классификации патчей
        self.final = nn.Conv2d(feature_maps*8, 1, kernel_size=4, stride=1, padding=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x
