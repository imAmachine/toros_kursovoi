import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.gan.train.trainers import DiscriminatorModelTrainer, GeneratorModelTrainer
from src.datasets.dataset import IceRidgeDataset
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


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
        train_dataset = IceRidgeDataset(train_metadata, dataset_processor=self.dataset.processor, with_target=True)
        val_dataset = IceRidgeDataset(val_metadata, dataset_processor=self.dataset.processor, with_target=False)
        
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

        if self.load_weights:
            try:
                self._load_weights()
                print("Веса моделей загружены успешно")
            except Exception as e:
                print(f"Ошибка загрузки весов: {e}")
        
        # Основной цикл обучения
        for epoch in range(self.epochs):
            self.model.generator.train()
            self.model.discriminator.train()
            
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            epoch_g_losses = {"total_loss": 0.0}
            epoch_d_losses = {"total_loss": 0.0}
            
            for inputs, targets, masks in progress:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                # Обучение генератора
                g_loss_dict, fake_images = self.g_trainer.train_pipeline_step(inputs, masks, targets)
                
                # Обучение дискриминатора
                d_loss_dict = self.d_trainer.train_pipeline_step(targets, fake_images, masks)

                # Обновление суммы потерь за эпоху
                for key, value in g_loss_dict.items():
                    epoch_g_losses[key] = epoch_g_losses.get(key, 0.0) + value
                
                for key, value in d_loss_dict.items():
                    epoch_d_losses[key] = epoch_d_losses.get(key, 0.0) + value
                
                progress.set_postfix({
                    "G_loss": g_loss_dict["total_loss"], 
                    "D_loss": d_loss_dict["total_loss"]
                })
            
            # Средние значения loss за эпоху
            for key in epoch_g_losses:
                epoch_g_losses[key] /= len(train_loader)
            
            for key in epoch_d_losses:
                epoch_d_losses[key] /= len(train_loader)
            
            print(f"Epoch {epoch+1}/{self.epochs} - G_loss: {epoch_g_losses['total_loss']:.4f}, D_loss: {epoch_d_losses['total_loss']:.4f}")
            self._save_models()
            
            # Валидация и визуализация результатов
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
                    composite, generated_val = self.model.generator(val_inputs, val_masks)
                    
                    # Визуализация
                    plt.figure(figsize=(15, 15))
                    for i in range(min(5, len(val_inputs))):  # Первые 5 изображений
                        # Исходное изображение с маской
                        plt.subplot(4, 5, i + 1)
                        plt.imshow(val_inputs[0].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Input [Batch {batch_idx}]')
                        plt.axis('off')
                        
                        # Маска
                        plt.subplot(4, 5, i + 6)
                        plt.imshow(val_masks[0].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Mask [Batch {batch_idx}]')
                        plt.axis('off')
                        
                        # Сгенерированное изображение
                        plt.subplot(4, 5, i + 11)
                        plt.imshow(generated_val[0].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Generated [Batch {batch_idx}]')
                        plt.axis('off')
                        
                        # Целевое изображение
                        plt.subplot(4, 5, i + 16)
                        plt.imshow(val_targets[0].cpu().squeeze().numpy(), cmap='gray')
                        plt.title(f'Target [Batch {batch_idx}]')
                        plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_path, f'samples_epoch.png'), dpi=300)
                    plt.close()
            
        return self.g_trainer.loss_history, self.d_trainer.loss_history