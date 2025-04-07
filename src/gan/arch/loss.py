import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gan.arch.common import VGG19, gaussian_blur


class L1:
    def __init__(
        self,
    ):
        self.calc = torch.nn.L1Loss()

    def __call__(self, x, y):
        return self.calc(x, y)


class Perceptual(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(x_vgg[f"relu{prefix[i]}_1"], y_vgg[f"relu{prefix[i]}_1"])
        return content_loss


class Style(nn.Module):
    def __init__(self):
        super(Style, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(
                self.compute_gram(x_vgg[f"relu{pre}_{pos}"]), self.compute_gram(y_vgg[f"relu{pre}_{pos}"])
            )
        return style_loss


class nsgan:
    def __init__(
        self,
    ):
        self.loss_fn = torch.nn.Softplus()

    def __call__(self, netD, fake, real):
        fake_detach = fake.detach()
        d_fake = netD(fake_detach)
        d_real = netD(real)
        dis_loss = self.loss_fn(-d_real).mean() + self.loss_fn(d_fake).mean()

        g_fake = netD(fake)
        gen_loss = self.loss_fn(-g_fake).mean()

        return dis_loss, gen_loss


class smgan:
    def __init__(self, ksize=71):
        self.ksize = ksize
        self.loss_fn = nn.MSELoss()

    def __call__(self, netD, fake, real, masks):
        fake_detach = fake.detach()

        g_fake = netD(fake)
        d_fake = netD(fake_detach)
        d_real = netD(real)

        _, _, h, w = g_fake.size()
        b, c, ht, wt = masks.size()

        # Handle inconsistent size between outputs and masks
        if h != ht or w != wt:
            g_fake = F.interpolate(g_fake, size=(ht, wt), mode="bilinear", align_corners=True)
            d_fake = F.interpolate(d_fake, size=(ht, wt), mode="bilinear", align_corners=True)
            d_real = F.interpolate(d_real, size=(ht, wt), mode="bilinear", align_corners=True)
        d_fake_label = gaussian_blur(masks, (self.ksize, self.ksize), (10, 10)).detach().cuda()
        d_real_label = torch.zeros_like(d_real).cuda()
        g_fake_label = torch.ones_like(g_fake).cuda()

        dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(d_real, d_real_label)
        gen_loss = self.loss_fn(g_fake, g_fake_label) * masks / torch.mean(masks)

        return dis_loss.mean(), gen_loss.mean()


class FractalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake, target, mask):
        b, c, h, w = fake.shape
        loss = 0.0
        for i in range(b):
            fake_region = fake[i].mean(dim=0) * mask[i, 0]
            target_region = target[i].mean(dim=0) * mask[i, 0]

            fd_fake = self.box_count(fake_region)
            fd_target = self.box_count(target_region)

            loss += abs(fd_fake - fd_target)
        return loss / b

    def box_count(self, img, threshold=0.5):
        sizes = [2, 4, 8, 16, 32]
        counts = []

        img_np = img.detach().cpu().numpy()
        img_np = (img_np > threshold).astype(np.uint8)

        for size in sizes:
            count = 0
            for i in range(0, img_np.shape[0], size):
                for j in range(0, img_np.shape[1], size):
                    patch = img_np[i:i + size, j:j + size]
                    if patch.any():
                        count += 1
            counts.append(count)

        log_sizes = np.log(np.array(sizes))
        log_counts = np.log(np.array(counts) + 1e-8)

        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return -coeffs[0]