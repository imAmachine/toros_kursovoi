import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.datasets.dataset import IceRidgeDataset
from .gan_arch import GANModel
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np


class GANTrainer:
    def __init__(self, model: GANModel, train_examples, val_examples, output_path, 
                epochs, batch_size, lr_g, lr_d, noise_type='gaussian', noise_level=0.05, 
                load_weights=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Параметры данных
        self.train_examples = train_examples
        self.val_examples = val_examples
        
        # Параметры обучения и шума
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.noise_type = noise_type
        self.noise_level = noise_level
        
        # Остальной код без изменений
        self.model = model
        self.load_weights = load_weights
        
        # Создаем директорию
        os.makedirs(self.output_path, exist_ok=True)
        
        # Инициализация параметров модели
        self._init_model_params()
        
        # Подготовка данных
        self._prepare_data()

    def _init_model_params(self):
        if self.load_weights: 
            self._load_weights()
        
        # объявление оптимизаторов
        self.optimizer_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr_g)
        self.optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.lr_d)
        
        # Инициализация планировщиков без параметра verbose
        self.scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_g, mode='min', factor=0.5, patience=5)
        self.scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_d, mode='min', factor=0.5, patience=5)
        
        # объявление loss функций
        self.loss_fn_g = smp.losses.DiceLoss(mode='binary')
        self.loss_fn_d = torch.nn.BCEWithLogitsLoss()

    def _load_weights(self):
        os.makedirs(self.output_path, exist_ok=True)
        
        gen_path = os.path.join(self.output_path, 'generator.pth')
        discr_path = os.path.join(self.output_path, 'discriminator.pth')
        if os.path.exists(gen_path) and os.path.exists(discr_path):
            self.model.load_weights(os.path.join(self.output_path, 'generator.pth'),
                                    os.path.join(self.output_path, 'discriminator.pth'))
        else:
            assert 'Ошибка загрузки весов моделей'
    
    def _prepare_data(self):
        dataset = IceRidgeDataset(
            examples=self.train_examples,
            transform=self.model.mask_transform,
            noise_type=self.noise_type,
            noise_level=self.noise_level
        )
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = IceRidgeDataset(
            examples=self.val_examples,
            transform=self.model.mask_transform,
            noise_type=self.noise_type,
            noise_level=self.noise_level
        )
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

    def _visualize_samples(self, epoch, sample_noised, sample_real, sample_mask, generated_mask):
        """Визуализация примеров в конце эпохи"""
        plt.figure(figsize=(15, 10))
        
        # Преобразуем тензоры в numpy массивы и убираем batch dimension
        sample_noised = sample_noised[0].cpu().numpy().transpose(1, 2, 0)
        sample_real = sample_real[0].cpu().numpy().transpose(1, 2, 0)
        sample_mask = sample_mask[0].cpu().numpy().squeeze()
        generated_mask = generated_mask[0].detach().cpu().numpy().squeeze()
        
        # Если изображение одноканальное, повторим по каналам для визуализации
        if sample_noised.shape[-1] == 1:
            sample_noised = np.repeat(sample_noised, 3, axis=-1)
        if sample_real.shape[-1] == 1:
            sample_real = np.repeat(sample_real, 3, axis=-1)
        
        plt.subplot(2, 2, 1)
        plt.imshow(sample_noised)
        plt.title('Noisy Input')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(sample_real)
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(sample_mask, cmap='gray')
        plt.title('Original Mask')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(generated_mask, cmap='gray')
        plt.title('Generated Mask')
        plt.axis('off')
        
        plt.suptitle(f'Epoch {epoch + 1}')
        plt.tight_layout()
        
        # Сохраняем изображение
        # save_path = os.path.join(self.output_path, f'epoch_{epoch+1}_visualization.png')
        # plt.savefig(save_path)
        # plt.close()
        plt.show()  # Раскомментируйте, если хотите видеть изображения во время обучения

    def train(self):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            progress_bar = tqdm(self.dataloader, desc=f"Training", total=len(self.dataloader))

            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            
            # Сохраним один batch для визуализации
            viz_batch = None

            for batch_idx, (input_mask, target_mask) in enumerate(self.dataloader):
                input_mask = input_mask.to(self.device)
                target_mask = target_mask.to(self.device)
                
                # Сохраняем первый batch для визуализации
                if batch_idx == 0:
                    viz_batch = (input_mask.clone(), target_mask.clone())

                # Генерация данных
                generated_mask = self.model.generator(input_mask)
                
                # Обучение дискриминатора
                real_output = self.model.discriminator(target_mask)
                fake_output = self.model.discriminator(generated_mask.detach())

                real_labels = torch.ones_like(real_output) * 0.9  # label smoothing
                fake_labels = torch.zeros_like(fake_output)

                loss_d_real = self.loss_fn_d(real_output, real_labels)
                loss_d_fake = self.loss_fn_d(fake_output, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) * 0.5

                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

                # Обучение генератора
                fake_output_g = self.model.discriminator(generated_mask)
                loss_g_adv = self.loss_fn_d(fake_output_g, torch.ones_like(fake_output_g))

                # Сохранение структуры исходной маски
                loss_g_preserve = F.mse_loss(generated_mask, target_mask)
                loss_g = loss_g_adv + 10.0 * loss_g_preserve

                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

                # Суммируем потери для планировщика
                epoch_loss_d += loss_d.item()
                epoch_loss_g += loss_g.item()

                progress_bar.set_postfix({
                    "D_loss": f"{loss_d.item():.4f}",
                    "G_loss": f"{loss_g.item():.4f}",
                    "Preserve": f"{loss_g_preserve.item():.4f}"
                })
                progress_bar.update()

                torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
                
                # Визуализация после эпохи
                if viz_batch is not None:
                    input_viz, target_viz = viz_batch
                    with torch.no_grad():
                        generated_viz = self.model.generator(input_viz)
                    self._visualize_samples(epoch, input_viz, target_viz, target_viz, generated_viz)
            
            progress_bar.close()
            
            # Средние потери за эпоху
            epoch_loss_g /= len(self.dataloader)
            epoch_loss_d /= len(self.dataloader)
            
            print(f"Epoch {epoch+1} - D_loss: {epoch_loss_d:.4f}, G_loss: {epoch_loss_g:.4f}")

            # Шаг планировщиков
            self.scheduler_g.step(epoch_loss_g)
            self.scheduler_d.step(epoch_loss_d)
            
            self._save_models()

    def _save_models(self):
        torch.save(self.model.generator.state_dict(), os.path.join(self.output_path, "generator.pth"))
        torch.save(self.model.discriminator.state_dict(), os.path.join(self.output_path, "discriminator.pth"))