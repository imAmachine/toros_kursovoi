import os
from typing import Dict
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.gan.model import GenerativeModel
from src.datasets.dataset import DatasetCreator


class GANTrainer:
    def __init__(self, model: GenerativeModel, dataset_processor: DatasetCreator, output_path, epochs=10, batch_size=8, load_weights=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.load_weights = load_weights
        self.dataset_processor = dataset_processor

        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.epoch_g_losses = {"total_loss": 0.0}
        self.epoch_d_losses = {"total_loss": 0.0}
        
        os.makedirs(self.output_path, exist_ok=True)
    
    def train(self):
        train_loader, val_loader = self.dataset_processor.get_dataloaders(batch_size=self.batch_size, 
                                                                          shuffle=True, 
                                                                          workers=4)
        
        if self.load_weights:
            try:
                self.model._load_weights(self.output_path)
                print("Веса моделей загружены успешно")
            except Exception as e:
                print(f"Ошибка загрузки весов: {e}")

        # Основной цикл обучения
        for epoch in range(self.epochs):
            self.model.generator.train()
            self.model.discriminator.train()

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for damaged, originals, damaged_masks in progress:
                damaged = damaged.to(self.device).detach()
                originals = originals.to(self.device).detach()
                damaged_masks = damaged_masks.to(self.device).detach()
                
                losses = self.model.train_step(inputs=damaged,
                                      targets=originals,
                                      masks=damaged_masks)
                
                self._calc_epoch_losses(losses)
                
                progress.set_postfix({
                    "G_loss": losses.get('g_losses').get('total_loss'), 
                    "D_loss": losses.get('d_losses').get('total_loss')
                })
            
            self._calc_avg_losses(batch_size=len(train_loader))
            
            self.model._save_models(self.output_path)
            
            val_g_loss_total = 0.0
            # Валидация и визуализация результатов
            with torch.no_grad():
                self.model.generator.eval()
                self.model.discriminator.eval()
                
                for batch_idx, (val_inputs, val_targets, val_masks) in enumerate(val_loader):
                    if batch_idx == 1:
                        break
                    
                    val_inputs = val_inputs.to(self.device).detach()
                    val_targets = val_targets.to(self.device).detach()
                    val_masks = val_masks.to(self.device).detach()

                    # Генерация изображений
                    generated_val = self.model.generator(val_inputs, val_masks)

                    # loss_dict_val, generated_val = self.model.g_trainer.val_pipeline_step(val_inputs, val_targets, val_masks)

                    # self.model.g_trainer.scheduler.step(loss_dict_val.get('total_loss'))
                    # self.model.d_trainer.scheduler.step(epoch_d_loss)
                    
                    # Визуализация
                    plt.figure(figsize=(15, 15))
                    for i in range(min(5, len(val_inputs))):  # Первые 5 изображений
                        # Исходное изображение с маской
                        plt.subplot(4, 5, i + 1)
                        plt.imshow(val_inputs[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Input [Batch {batch_idx}]')
                        plt.axis('off')
                        
                        # Маска
                        plt.subplot(4, 5, i + 6)
                        plt.imshow(val_masks[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Mask [Batch {batch_idx}]')
                        plt.axis('off')
                        
                        # Сгенерированное изображение
                        plt.subplot(4, 5, i + 11)
                        plt.imshow(generated_val[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Generated [Batch {batch_idx}]')
                        plt.axis('off')
                        
                        # Целевое изображение
                        plt.subplot(4, 5, i + 16)
                        plt.imshow(val_targets[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Target [Batch {batch_idx}]')
                        plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_path, f'samples_epoch.png'), dpi=300)
                    plt.close()

            print(f"Epoch {epoch+1}/{self.epochs} - G_loss: {self.epoch_g_losses.get('total_loss'):.4f}, D_loss: {self.epoch_d_losses.get('total_loss'):.4f}")

        self.model._save_models(self.output_path)
            
        return self.model.g_trainer.loss_history, self.model.d_trainer.loss_history
    
    def _calc_epoch_losses(self, losses: Dict):
        # Обновление суммы потерь за эпоху
        for key, value in losses['g_losses'].items():
            self.epoch_g_losses[key] = self.epoch_g_losses.get(key, 0.0) + value
        
        for key, value in losses['d_losses'].items():
            self.epoch_d_losses[key] = self.epoch_d_losses.get(key, 0.0) + value
    
    def _calc_avg_losses(self, batch_size):
        # Средние значения loss за эпоху
        for key in self.epoch_g_losses:
            self.epoch_g_losses[key] /= batch_size
        
        for key in self.epoch_d_losses:
            self.epoch_d_losses[key] /= batch_size