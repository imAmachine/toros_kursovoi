import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class BaseNetwork(torch.nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            "Network [%s] was created. Total number of parameters: %.1f million. "
            "To see the architecture, do print(network)." % (type(self).__name__, num_params / 1000000)
        )

    def init_weights(self, init_type="normal", gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("InstanceNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    torch.nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == "kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "none":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
                if hasattr(m, "bias") and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, gain)

class InpaintGenerator(BaseNetwork):
    def __init__(self, args):  # 1046
        super(InpaintGenerator, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(2, 64, 7),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 256, 4, stride=2, padding=1),
            torch.nn.ReLU(True),
        )

        self.middle = torch.nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])

        self.decoder = torch.nn.Sequential(
            UpConv(256, 128), torch.nn.ReLU(True), UpConv(128, 64), torch.nn.ReLU(True), torch.nn.Conv2d(64, 1, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(torch.nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = torch.nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class AOTBlock(torch.nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(rate), torch.nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), torch.nn.ReLU(True)
                ),
            )
        self.fuse = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

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
