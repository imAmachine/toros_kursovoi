import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.gan.train.trainers import DiscriminatorModelTrainer, GeneratorModelTrainer
from src.datasets.dataset import IceRidgeDataset
from src.gan.arch.gan import GANModel
from src.analyzer.fractal_funcs import FractalAnalyzer
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

def _calculate_dice_score(real, generated):
    dice_scores = []
    for pred, gt in zip(generated, real):
        pred_np = pred.squeeze().cpu().detach().numpy().round().astype(np.uint8)
        gt_np = gt.squeeze().cpu().numpy().round().astype(np.uint8)
        
        intersection = (pred_np & gt_np).sum()
        dice = (2. * intersection) / (pred_np.sum() + gt_np.sum() + 1e-6)
        dice_scores.append(dice)
    return sum(dice_scores) / len(dice_scores)

def _calculate_iou(real, generated):
    iou_scores = []
    for pred, gt in zip(generated, real):
        pred_np = pred.squeeze().cpu().detach().numpy().round().astype(np.uint8)
        gt_np = gt.squeeze().cpu().numpy().round().astype(np.uint8)
        
        intersection = (pred_np & gt_np).sum()
        union = (pred_np | gt_np).sum()
        iou = intersection / (union + 1e-6)
        iou_scores.append(iou)
    return sum(iou_scores) / len(iou_scores)

def _calculate_ssim(real, generated):
    ssim_scores = []
    for pred, gt in zip(generated, real):
        pred_np = pred.squeeze().cpu().detach().numpy().astype(np.float32)
        gt_np = gt.squeeze().cpu().numpy().astype(np.float32)
        score = ssim(gt_np, pred_np, data_range=1.0)
        ssim_scores.append(score)
    return sum(ssim_scores) / len(ssim_scores)


class GANTrainer:
    def __init__(self, model, dataset, output_path, epochs=10, batch_size=8, load_weights=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.load_weights = load_weights
        
        # модели тренеров
        self.g_trainer = GeneratorModelTrainer(
            model=model.generator,
            discriminator=model.discriminator,
            lambda_l1=0.1,
            lambda_perceptual=0.1,
            lambda_style=1,
            lambda_fd=1
        )
        
        self.d_trainer = DiscriminatorModelTrainer(
            model=model.discriminator
        )
        
        self.dataset = dataset
        
        # Параметры
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Создаем директорию
        os.makedirs(self.output_path, exist_ok=True)

    def _load_weights(self):
        os.makedirs(self.output_path, exist_ok=True)
        
        gen_path = os.path.join(self.output_path, 'generator.pt')
        discr_path = os.path.join(self.output_path, 'discriminator.pt')
        if os.path.exists(gen_path) and os.path.exists(discr_path):
            self.model.load_weights(gen_path, discr_path)
        else:
            raise FileNotFoundError('Ошибка загрузки весов моделей')
    
    def _save_models(self):
        self.g_trainer.save_model_state_dict(self.output_path)
        self.d_trainer.save_model_state_dict(self.output_path)

    def prepare_dataloaders(self):
        train_metadata, val_metadata = self.dataset.split_dataset(self.dataset.metadata, val_ratio=0.2)
        train_dataset = IceRidgeDataset(train_metadata, dataset_processor=self.dataset.processor)
        val_dataset = IceRidgeDataset(val_metadata, dataset_processor=self.dataset.processor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def train(self):
        train_loader, val_loader = self.prepare_dataloaders()
        
        # Основной цикл обучения
        for epoch in range(self.epochs):
            self.model.generator.train()
            self.model.discriminator.train()
            
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            for inputs, targets, masks in progress:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                # Обучение генератора
                g_loss, fake_images = self.g_trainer.train_pipeline_step(inputs, masks, targets)
                
                # Обучение дискриминатора
                d_loss = self.d_trainer.train_pipeline_step(targets, fake_images, masks)

                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                
                progress.set_postfix({"G_loss": g_loss, "D_loss": d_loss})
            
            # Средние значения loss за эпоху
            epoch_g_loss /= len(train_loader)
            epoch_d_loss /= len(train_loader)
            
            print(f"Epoch {epoch+1}/{self.epochs} - G_loss: {epoch_g_loss:.4f}, D_loss: {epoch_d_loss:.4f}")
            self._save_models()
            
            
            # нужно убрать этот страх===============================================================================
            with torch.no_grad():
                self.model.generator.eval()
                self.model.discriminator.eval()
                
                for batch_idx, (val_inputs, val_targets, val_masks) in enumerate(val_loader):
                    if batch_idx == 1:
                        break
                    
                    val_inputs = val_inputs.to(self.device)
                    val_targets = val_targets.to(self.device)
                    val_masks = val_masks.to(self.device)
                    
                    # Генерация изображений
                    generated_val = self.model.generator(val_inputs, val_masks)
                    
                    # Визуализация
                    plt.figure(figsize=(15, 10))
                    for i in range(min(5, len(val_inputs))):  # Первые 5 изображений
                        # Исходное изображение
                        plt.subplot(3, 5, i + 1)
                        plt.imshow(val_inputs[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Input [Batch {batch_idx}]')
                        plt.axis('off')
                        
                        # Сгенерированное изображение
                        plt.subplot(3, 5, i + 6)
                        plt.imshow(generated_val[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Generated [Batch {batch_idx}]')
                        plt.axis('off')
                        
                        # Целевое изображение
                        plt.subplot(3, 5, i + 11)
                        plt.imshow(val_targets[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Target [Batch {batch_idx}]')
                        plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_path, 'samples.png'), dpi=300)
                    plt.close()
        
        return self.g_trainer.loss_history, self.d_trainer.loss_history