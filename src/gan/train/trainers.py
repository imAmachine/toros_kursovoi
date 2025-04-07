import os
import torch
import torch.nn as nn
from src.gan.arch.gan_components import AOTDiscriminator, AOTGenerator
from .interfaces.model_trainer import IModelTrainer
import matplotlib.pyplot as plt
from src.gan.arch.loss import FractalLoss, Perceptual, Style, smgan

class GeneratorModelTrainer(IModelTrainer):
    def __init__(self, model, discriminator, optimizer=None, lambda_l1=1, lambda_perceptual=1, lambda_style=1, lambda_fd=1):
        self.model = model
        self.discriminator = discriminator
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = Perceptual()
        self.style_loss = Style()
        self.adv_loss = smgan(ksize=71)
        self.fractal_loss = FractalLoss()

        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        self.lambda_fd = lambda_fd

        self.loss_history = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "generator.pt"))
    
    def train_pipeline_step(self, inputs, masks, targets, fd=None):
        self.optimizer.zero_grad()

        fake_images = self.model(inputs, masks)
        comp_images = masks * inputs + masks * fake_images
        
        # Correct loss calculations
        # Use element-wise multiplication instead of boolean indexing
        losses = {}
        losses['l1'] = self.l1_loss(fake_images, targets) * self.lambda_l1
        losses['perceptual'] = self.perceptual_loss(fake_images, targets) * self.lambda_perceptual
        losses['style'] = self.style_loss(fake_images, targets) * self.lambda_style
        losses['fd'] = self.fractal_loss(fake_images, targets, masks) * self.lambda_fd
        
        _, gen_loss = self.adv_loss(self.discriminator, comp_images, targets, masks)
        losses['adv'] = gen_loss * 0.5
        
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()

        self.loss_history.append(total_loss.item())
        return total_loss.item(), fake_images

        

class DiscriminatorModelTrainer(IModelTrainer):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.adv_loss = smgan(ksize=71)
        self.loss_history = []

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "dicriminator.pt"))
    
    def train_pipeline_step(self, targets, fake_images, masks):
        self.optimizer.zero_grad()
        
        # Вычисление потерь дискриминатора
        dis_loss, _ = self.adv_loss(self.model, fake_images.detach(), targets, masks)
        
        dis_loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(dis_loss.item())
        return dis_loss.item()