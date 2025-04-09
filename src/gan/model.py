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
        self.discriminator = GanDiscriminator(input_channels=2, feature_maps=d_feature_maps).to(self.device)
        
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
        g_loss_dict, fake_images = self.g_trainer.train_pipeline_step(inputs, targets, masks)
        d_loss_dict = self.d_trainer.train_pipeline_step(inputs, targets, fake_images)
        
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

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=10, verbose=True)

        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.loss_history = []
        self.loss_history_val = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "generator.pt"))

    def _calc_adv_loss(self, x_input, x_generated):
        fake_pred = self.discriminator(x_input, x_generated)
        real_label = torch.ones_like(fake_pred)
        return self.adv_criterion(fake_pred, real_label)
    
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
        return torch.tensor(fd_loss, device=generated.device)

    def train_pipeline_step(self, input_masked, target, mask):
        device = next(self.model.parameters()).device

        input_masked = input_masked.to(device)
        target = target.to(device)
        mask = mask.to(device)

        self.optimizer.zero_grad()

        generated = self.model(input_masked, mask)

        # (дискриминатор должен принять вход + результат генератора)
        adv_loss = self._calc_adv_loss(input_masked, generated)

        # L1 loss только по маскированной области
        l1_loss = torch.abs(generated - target) * mask
        l1_loss = l1_loss.mean()

        # Расчёт разницы фрактальной размерности
        fd_loss = self._calc_fractal_loss(generated, mask)

        # Общий генераторный loss
        total_loss = adv_loss + l1_loss + fd_loss

        total_loss.backward()
        self.optimizer.step()

        loss_dict = {
            'adv_loss': adv_loss.item(),
            'l1_loss': l1_loss.item(),
            'fd_loss': fd_loss.item(),
            'total_loss': total_loss.item()
        }
        self.loss_history.append(loss_dict)

        return loss_dict, generated
    
    def val_pipeline_step(self, val_inputs, val_targets, val_masks):

        generated_val = self.model(val_inputs, val_masks)

        # (дискриминатор должен принять вход + результат генератора)
        adv_loss_val = self._calc_adv_loss(val_masks, generated_val)

        # L1 loss только по маскированной области
        l1_loss_val = torch.abs(generated_val - val_targets) * val_masks
        l1_loss_val = l1_loss_val.mean()

        # Расчёт разницы фрактальной размерности
        fd_loss_val = self._calc_fractal_loss(generated_val, val_masks)

        # Общий генераторный loss
        total_loss_val = adv_loss_val + l1_loss_val + fd_loss_val

        loss_dict_val = {
            'adv_loss': adv_loss_val.item(),
            'l1_loss': l1_loss_val.item(),
            'fd_loss': fd_loss_val.item(),
            'total_loss': total_loss_val.item()
        }
        self.loss_history_val.append(loss_dict_val)

        return loss_dict_val, generated_val


class DiscriminatorModelTrainer(IModelTrainer):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=7, verbose=True)
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss_history = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "discriminator.pt"))

    def _calc_adv_loss(self, fake_input, fake_output, real_input, real_output):
        real_pred = self.model(real_input, real_output)
        fake_pred = self.model(fake_input, fake_output.detach())

        real_label = torch.ones_like(real_pred)
        fake_label = torch.zeros_like(fake_pred)

        real_loss = self.criterion(real_pred, real_label)
        fake_loss = self.criterion(fake_pred, fake_label)

        return {'real_loss': real_loss, 'fake_loss': fake_loss}

    def train_pipeline_step(self, input_masked, real_target, fake_generated):
        device = next(self.model.parameters()).device

        input_masked = input_masked.to(device)
        real_target = real_target.to(device)
        fake_generated = fake_generated.to(device)

        self.optimizer.zero_grad()
        adv_losses = self._calc_adv_loss(
            fake_input=input_masked, fake_output=fake_generated,
            real_input=input_masked, real_output=real_target
        )
        disc_loss = 0.5 * (adv_losses['real_loss'] + adv_losses['fake_loss'])

        disc_loss.backward()
        self.optimizer.step()

        loss_dict = {
            'total_loss': disc_loss.item()
        }
        self.loss_history.append(loss_dict)

        return loss_dict
