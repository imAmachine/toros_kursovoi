import torch

class GatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels*2, kernel_size, stride, padding)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, mask):
        out = self.conv(x)
        features, gate = torch.split(out, out.shape[1]//2, dim=1)
        gate = self.sigmoid(gate + mask)
        return features * gate
    
class AOTGenerator(torch.nn.Module):
    def __init__(self, input_channels=2, feature_maps=64):
        super().__init__()
        # Encoder
        self.enc1 = GatedConv(input_channels, feature_maps, kernel_size=7, padding=3)
        self.enc2 = GatedConv(feature_maps, feature_maps*2, kernel_size=4, stride=2, padding=1)
        
        # AOT Blocks
        self.aot_block = torch.nn.Sequential(
            torch.nn.Conv2d(feature_maps*2, feature_maps*2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(feature_maps*2, feature_maps*2, kernel_size=3, padding=1)
        )
        
        # Decoder
        self.dec1 = torch.nn.ConvTranspose2d(feature_maps*2, feature_maps, kernel_size=4, stride=2, padding=1)
        self.dec2 = GatedConv(feature_maps, 1, kernel_size=7, padding=3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.enc1(x, mask)
        x = self.enc2(x, mask)
        x = x + self.aot_block(x)
        x = self.dec1(x)
        x = self.dec2(x, mask)
        return self.sigmoid(x)