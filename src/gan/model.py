import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.analyzer.fractal_funcs import FractalAnalyzer, FractalAnalyzerGPU

from src.gan.gan_arch import GanDiscriminator, GanGenerator
from .interfaces import IModelTrainer

class GenerativeModel:
    def __init__(self, target_image_size=512, g_feature_maps=64, d_feature_maps=16):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = GanGenerator(input_channels=2, feature_maps=g_feature_maps).to(self.device)
        self.discriminator = GanDiscriminator(input_channels=1, feature_maps=d_feature_maps).to(self.device)
        
        self.g_trainer, self.d_trainer = self._init_trainers()
        
        self.target_image_size = target_image_size

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            # transforms.GaussianBlur(1),
            transforms.Resize((self.target_image_size, self.target_image_size)),
            transforms.ToTensor()
        ])

    def _init_trainers(self):
        g_trainer = GeneratorModelTrainer(model=self.generator,discriminator=self.discriminator)
        d_trainer = DiscriminatorModelTrainer(model=self.discriminator)
        
        return g_trainer, d_trainer
    
    def train_step(self, inputs, targets, masks):
        g_loss_dict, fake_images = self.g_trainer.step(inputs, targets, masks)        
        d_loss_dict = self.d_trainer.step(targets, fake_images, masks)
        
        return {'g_losses': g_loss_dict, 'd_losses': d_loss_dict}
    
    def _save_models(self, output_path):
        self.g_trainer.save_model_state_dict(output_path)
        self.d_trainer.save_model_state_dict(output_path)

    def _load_weights(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        
        gen_path = os.path.join(output_path, 'generator.pt')
        discr_path = os.path.join(output_path, 'discriminator.pt')
        if os.path.exists(gen_path) and os.path.exists(discr_path):
            self.generator.load_state_dict(torch.load(gen_path, map_location=self.device, weights_only=True))
            self.discriminator.load_state_dict(torch.load(discr_path, map_location=self.device, weights_only=True))
        else:
            raise FileNotFoundError('Ошибка загрузки весов моделей')


class GeneratorModelTrainer(IModelTrainer):
    def __init__(self, model, discriminator):
        self.model = model
        self.discriminator = discriminator
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.5, 0.999))

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=10, verbose=True)

        self.adv_criterion = nn.BCELoss()
        self.loss_history = []
        self.loss_history_val = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "generator.pt"))

    def _calc_adv_loss(self, target, generated, mask):
        fake_pred_known = self.discriminator(generated * (1 - mask))
        real_label_known = torch.ones_like(fake_pred_known)
        
        fake_pred_damaged = self.discriminator(generated * mask)
        real_label_damaged = torch.ones_like(fake_pred_known)
        
        total_adv_loss = (self.adv_criterion(fake_pred_known, real_label_known) + self.adv_criterion(fake_pred_damaged, real_label_damaged)) * 0.5
        
        return total_adv_loss
    
    def _calc_fractal_loss(self, generated, masks):
        fd_losses = 0
        batch_size = generated.shape[0]
        for i in range(batch_size):
            img = generated[i].squeeze()
            m = masks[i].squeeze()

            part_masked = img * m
            part_unmasked = img * (1 - m)

            fd_masked = FractalAnalyzerGPU.calculate_fractal_dimension(
                *FractalAnalyzerGPU.box_counting(part_masked),
                device=generated.device
            )
            fd_unmasked = FractalAnalyzerGPU.calculate_fractal_dimension(
                *FractalAnalyzerGPU.box_counting(part_unmasked),
                device=generated.device
            )

            fd_losses += abs(fd_masked - fd_unmasked)
        fd_loss = fd_losses / batch_size
        return fd_loss

    def step(self, input_masked, target, mask):
        self.optimizer.zero_grad()
        generated = self.model(input_masked, mask)

        adv_loss = self._calc_adv_loss(target, generated, mask)
        l1_loss = torch.abs(target - generated) * mask
        l1_loss = l1_loss.mean()
        fd_loss = 0#self._calc_fractal_loss(generated, mask)

        # Общий генераторный loss
        total_loss = adv_loss + l1_loss + fd_loss

        total_loss.backward()
        self.optimizer.step()

        loss_dict = {
            'adv_loss': adv_loss.item(),
            'l1_loss': l1_loss.item(),
            'fd_loss': 0,#fd_loss.item(),
            'total_loss': total_loss.item()
        }
        self.loss_history.append(loss_dict)

        return loss_dict, generated


class DiscriminatorModelTrainer(IModelTrainer):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.5, 0.999))
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=7, verbose=True)
        self.criterion = nn.BCELoss()
        self.loss_history = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "discriminator.pt"))
    
    def _calc_adv_loss(self, real_target, fake_generated, masks):
        # Вычисление предсказаний для "реальных" данных
        real_pred_known = self.model(real_target * (1 - masks))
        real_label_known = torch.ones_like(real_pred_known)

        real_pred_damaged = self.model(real_target * masks)
        real_label_damaged = torch.ones_like(real_pred_damaged)

        # Вычисление предсказаний для "фейковых" данных
        fake_pred_known = self.model(fake_generated * (1 - masks))
        fake_label_known = torch.zeros_like(fake_pred_known)

        fake_pred_damaged = self.model(fake_generated * masks)
        fake_label_damaged = torch.zeros_like(fake_pred_damaged)

        # Суммируем лосс для реальных и фейковых данных
        real_loss = self.criterion(real_pred_known, real_label_known) + self.criterion(real_pred_damaged, real_label_damaged)
        fake_loss = self.criterion(fake_pred_known, fake_label_known) + self.criterion(fake_pred_damaged, fake_label_damaged)

        total_adv_loss = real_loss + fake_loss

        return total_adv_loss

    def step(self, real_target, fake_generated, masks):
        self.optimizer.zero_grad()
        total_adv_loss = self._calc_adv_loss(real_target=real_target, 
                                         fake_generated=fake_generated.detach(),
                                         masks=masks)

        total_adv_loss.backward()
        self.optimizer.step()

        loss_dict = {
            'total_loss': total_adv_loss.item()
        }
        self.loss_history.append(loss_dict)

        return loss_dict
