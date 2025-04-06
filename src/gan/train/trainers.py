import os
import torch
import torch.nn as nn
from src.gan.arch.gan_components import AOTDiscriminator
from src.gan.arch.gan_components import AOTGenerator
from .interfaces.model_trainer import IModelTrainer

class GeneratorModelTrainer(IModelTrainer):
    def __init__(self, model: AOTGenerator, discriminator: AOTDiscriminator, scheduler=None, 
                optimizer=None, loss_fn=None, lambda_l1=100.0):
        self.model = model
        self.discriminator = discriminator
        self.scheduler = scheduler
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        else:
            self.optimizer = optimizer
            
        if loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = loss_fn
            
        self.l1_loss = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.loss_history = []
    
    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "generator.pt"))
    
    def model_generate(self, inputs, masks):
        return self.model(inputs, masks)
    
    def train_pipeline_step(self, inputs, masks, targets):
        self.optimizer.zero_grad()
        
        # Генерируем фейковые изображения
        fake_images = self.model_generate(inputs, masks)
        
        # GAN loss (обмануть дискриминатор)
        fake_outputs = self.discriminator(fake_images)
        real_labels = torch.ones_like(fake_outputs, device=fake_outputs.device)
        g_loss_gan = self.loss_fn(fake_outputs, real_labels)
        
        # L1 loss (pixel-wise)
        g_loss_l1 = self.l1_loss(fake_images, targets) * self.lambda_l1
        
        # Общая loss генератора
        loss = g_loss_gan + g_loss_l1
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
            
        self.loss_history.append(loss.item())
        
        return loss.item(), fake_images
        

class DiscriminatorModelTrainer(IModelTrainer):
    def __init__(self, model: AOTDiscriminator, scheduler=None, optimizer=None, loss_fn=None):
        self.model = model
        self.scheduler = scheduler
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        else:
            self.optimizer = optimizer
            
        if loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = loss_fn
            
        self.loss_history = []
    
    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "discriminator.pt"))
    
    def model_classify(self, inputs):
        return self.model(inputs)
    
    def train_pipeline_step(self, targets, fake_images):
        self.optimizer.zero_grad()
        
        # Реальные изображения
        real_outputs = self.model_classify(targets)
        real_labels = torch.ones_like(real_outputs, device=real_outputs.device)
        loss_real = self.loss_fn(real_outputs, real_labels)
        
        # Сгенерированные изображения
        fake_outputs = self.model(fake_images.detach())
        fake_labels = torch.zeros_like(fake_outputs, device=fake_outputs.device)
        loss_fake = self.loss_fn(fake_outputs, fake_labels)
        
        # Общая loss дискриминатора
        loss = (loss_real + loss_fake) * 0.5
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
            
        self.loss_history.append(loss.item())
        
        return loss.item()