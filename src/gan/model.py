import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms

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
            transforms.GaussianBlur(1),
            transforms.Resize((self.target_image_size, self.target_image_size)),
            transforms.ToTensor()
        ])

    def _init_trainers(self):
        g_trainer = GeneratorModelTrainer(model=self.generator,discriminator=self.discriminator)
        d_trainer = DiscriminatorModelTrainer(model=self.discriminator)
        
        return g_trainer, d_trainer
    
    def train_step(self, inputs, targets, masks):
        g_loss_dict, fake_images = self.g_trainer.train_pipeline_step(inputs, masks)
        d_loss_dict = self.d_trainer.train_pipeline_step(targets, fake_images)
        
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
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
        
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.loss_history = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "generator.pt"))
    
    def _calc_adv_loss(self, generated_image, masks):
        fake_pred = self.discriminator(generated_image * (1 - masks))
        real_label = torch.ones_like(fake_pred)
        return self.adv_criterion(fake_pred, real_label)
    
    def _calc_adv_masked_region_loss(self, generated_image, mask):
        fake_pred = self.discriminator(generated_image * mask)
        real_label = torch.ones_like(fake_pred)
        return self.adv_criterion(fake_pred, real_label)
    
    def train_pipeline_step(self, inputs, masks):
        self.optimizer.zero_grad()
        generated = self.model(inputs, masks)
        
        gen_adv_loss = self._calc_adv_loss(generated, masks)
        gen_adv_loss_masked = self._calc_adv_masked_region_loss(generated, masks)
        
        gen_total_loss = gen_adv_loss + gen_adv_loss_masked * 1.1
        
        gen_total_loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            'adv_loss': gen_adv_loss.item(),
            'adv_masked_loss': gen_adv_loss_masked.item(),
            'total_loss': gen_total_loss.item()
        }
        self.loss_history.append(loss_dict)
        
        return loss_dict, generated


class DiscriminatorModelTrainer(IModelTrainer):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss_history = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "discriminator.pt"))
    
    def _calc_adv_loss(self, fake_images, real_images):
        real_pred = self.model(real_images)
        fake_pred = self.model(fake_images.detach())
        
        real_label = torch.ones_like(real_pred)
        fake_label = torch.zeros_like(fake_pred)
        
        real_loss = self.criterion(real_pred, real_label)
        fake_loss = self.criterion(fake_pred, fake_label)
        
        return {'real_loss': real_loss, 'fake_loss': fake_loss}
    
    def train_pipeline_step(self, targets, fake_images):
        # Перемещаем данные на устройство
        targets = targets.to(next(self.model.parameters()).device)
        fake_images = fake_images.to(next(self.model.parameters()).device)
        
        self.optimizer.zero_grad()
        adv_losses = self._calc_adv_loss(fake_images, targets)
        disc_loss = (adv_losses.get('real_loss') + adv_losses.get('fake_loss')) * 0.5
        
        disc_loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            'total_loss': disc_loss.item()
        }
        self.loss_history.append(loss_dict)
        
        return loss_dict