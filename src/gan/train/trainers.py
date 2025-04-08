import os
import torch
import torch.nn as nn
import torchvision
from .interfaces.model_trainer import IModelTrainer

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:23])

        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)


class GeneratorModelTrainer(IModelTrainer):
    def __init__(self, model, discriminator, optimizer=None, lambda_l1=100, lambda_perceptual=10, lambda_style=1, lambda_fd=1):
        self.model = model
        self.discriminator = discriminator
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.5, 0.999))
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        self.lambda_fd = lambda_fd
        
        # Создаем экстрактор признаков для perceptual loss если нужно
        self.feature_extractor = None
        if lambda_perceptual > 0 or lambda_style > 0:
            self.feature_extractor = FeatureExtractor().to(next(model.parameters()).device)
        
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()
        self.loss_history = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "generator.pt"))
    
    def _gram_matrix(self, features):
        batch_size, channels, height, width = features.size()
        features_flat = features.view(batch_size, channels, height * width)
        features_t = features_flat.transpose(1, 2)
        gram = features_flat.bmm(features_t) / (channels * height * width)
        return gram
    
    def train_pipeline_step(self, inputs, masks, targets, fd=None):
        # Перемещаем данные на устройство
        inputs = inputs.to(next(self.model.parameters()).device)
        masks = masks.to(next(self.model.parameters()).device)
        targets = targets.to(next(self.model.parameters()).device)
        
        # Обнуляем градиенты
        self.optimizer.zero_grad()
        
        # Генерируем изображение
        composite, generated = self.model(inputs, masks)
        
        masked_targets = targets * masks
        masked_generated = composite * (1 - masks)
        
        # Adversarial loss
        fake_pred = self.discriminator(masked_generated)
        real_label = torch.ones_like(fake_pred, device=fake_pred.device)
        gen_adv_loss = self.adv_criterion(fake_pred, real_label)
        
        # Удаить ошибки ---------------------------------------
        # # L1 loss
        # gen_l1_loss = self.l1_criterion(masked_generated, masked_targets) * self.lambda_l1
        
        # # Perceptual loss (если применимо)
        # gen_perceptual_loss = torch.tensor(0.0, device=inputs.device)
        # if self.lambda_perceptual > 0 and self.feature_extractor is not None:
        #     real_features = self.feature_extractor(masked_targets)
        #     fake_features = self.feature_extractor(masked_generated)
        #     gen_perceptual_loss = self.l1_criterion(fake_features, real_features) * self.lambda_perceptual
        
        # # Style loss (если применимо)
        # gen_style_loss = torch.tensor(0.0, device=inputs.device)
        # if self.lambda_style > 0 and self.feature_extractor is not None:
        #     real_features = self.feature_extractor(masked_targets)
        #     fake_features = self.feature_extractor(masked_generated)
        #     real_gram = self._gram_matrix(real_features)
        #     fake_gram = self._gram_matrix(fake_features)
        #     gen_style_loss = self.l1_criterion(fake_gram, real_gram) * self.lambda_style
        
        # # Feature distribution loss (если применимо)
        # gen_fd_loss = torch.tensor(0.0, device=inputs.device)
        # if self.lambda_fd > 0 and fd is not None:
        #     gen_fd_loss = fd(masked_generated, masked_targets) * self.lambda_fd
        
        # Общая потеря
        gen_total_loss = gen_adv_loss
        
        # Обратное распространение
        gen_total_loss.backward()
        self.optimizer.step()
        
        # Сохраняем историю потерь
        loss_dict = {
            'adv_loss': gen_adv_loss.item(),
            'total_loss': gen_total_loss.item()
        }
        self.loss_history.append(loss_dict)
        
        return loss_dict, generated

        

class DiscriminatorModelTrainer(IModelTrainer):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss_history = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "discriminator.pt"))
    
    def train_pipeline_step(self, targets, fake_images, masks):
        # Перемещаем данные на устройство
        targets = targets.to(next(self.model.parameters()).device)
        fake_images = fake_images.to(next(self.model.parameters()).device)
        
        # Обнуляем градиенты
        self.optimizer.zero_grad()
        
        # Реальное изображение
        real_pred = self.model(targets)
        real_label = torch.zeros_like(real_pred, device=real_pred.device)
        real_loss = self.criterion(real_pred, real_label)
        
        # Поддельное изображение
        fake_pred = self.model(fake_images.detach())  # .detach() для отрыва от графа вычислений генератора
        fake_label = torch.ones_like(fake_pred, device=fake_pred.device)
        fake_loss = self.criterion(fake_pred, fake_label)
        
        # Общая потеря
        disc_loss = (real_loss + fake_loss) * 0.5
        
        # Обратное распространение
        disc_loss.backward()
        self.optimizer.step()
        
        # Сохраняем историю потерь
        loss_dict = {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'total_loss': disc_loss.item()
        }
        self.loss_history.append(loss_dict)
        
        return loss_dict