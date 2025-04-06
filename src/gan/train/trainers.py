import os
import torch
import torch.nn as nn
from src.gan.arch.gan_components import AOTDiscriminator
from src.gan.arch.gan_components import InpaintGenerator
from .interfaces.model_trainer import IModelTrainer
import matplotlib.pyplot as plt
from src.gan.arch.loss import Perceptual, Style, smgan

class GeneratorModelTrainer:
    def __init__(self, model, discriminator, optimizer=None, 
                lambda_l1=1.0, lambda_perceptual=0.1, lambda_style=0.1):
        self.model = model
        self.discriminator = discriminator
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Инициализация loss-функций AOT-GAN
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = Perceptual()
        self.style_loss = Style()
        self.adv_loss = smgan(ksize=71)
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        
        self.loss_history = []

    def train_pipeline_step(self, inputs, masks, targets):
        self.optimizer.zero_grad()
        
        # Генерация изображения
        fake_images = self.model(inputs, masks)
        comp_images = (1 - masks) * inputs + masks * fake_images
        
        # Вычисление потерь
        losses = {}
        
        # Реконструкционные потери
        losses['l1'] = self.l1_loss(fake_images, targets) * self.lambda_l1
        losses['perceptual'] = self.perceptual_loss(fake_images, targets) * self.lambda_perceptual
        losses['style'] = self.style_loss(fake_images, targets) * self.lambda_style
        
        # Adversarial loss
        _, gen_loss = self.adv_loss(self.discriminator, comp_images, targets, masks)
        losses['adv'] = gen_loss
        
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(total_loss.item())
        return total_loss.item(), fake_images

        

class DiscriminatorModelTrainer:
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.adv_loss = smgan(ksize=71)
        self.loss_history = []

    def train_pipeline_step(self, targets, fake_images, masks):
        self.optimizer.zero_grad()
        
        # Вычисление потерь дискриминатора
        dis_loss, _ = self.adv_loss(self.model, fake_images.detach(), targets, masks)
        
        dis_loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(dis_loss.item())
        return dis_loss.item()