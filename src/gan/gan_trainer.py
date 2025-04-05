import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.datasets.dataset import IceRidgeDataset
from src.gan.arch.gan import GANModel
from src.analyzer.fractal_funcs import FractalAnalyzer
from tqdm import tqdm
import numpy as np


class GANTrainer:
    def __init__(self, model: GANModel, dataset: IceRidgeDataset, output_path, 
                epochs, batch_size, lr_g, lr_d, load_weights=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dataset = dataset
        # Параметры обучения и шума
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        
        self.model = model
        self.load_weights = load_weights
        
        # Создаем директорию
        os.makedirs(self.output_path, exist_ok=True)

    def _load_weights(self):
        os.makedirs(self.output_path, exist_ok=True)
        
        gen_path = os.path.join(self.output_path, 'generator.pth')
        discr_path = os.path.join(self.output_path, 'discriminator.pth')
        if os.path.exists(gen_path) and os.path.exists(discr_path):
            self.model.load_weights(os.path.join(self.output_path, 'generator.pth'),
                                    os.path.join(self.output_path, 'discriminator.pth'))
        else:
            assert 'Ошибка загрузки весов моделей'

    def _calculate_fractal_loss(self, real, generated):
        batch_loss = 0.0
        for gen, gt in zip(generated, real):
            # Конвертация тензоров в numpy массивы
            gen_np = gen.squeeze().cpu().detach().numpy().round().astype(np.uint8)
            gt_np = gt.squeeze().cpu().numpy().round().astype(np.uint8)
            
            # Расчет ФР
            sizes_gen, counts_gen = FractalAnalyzer.box_counting(gen_np)
            sizes_gt, counts_gt = FractalAnalyzer.box_counting(gt_np)
            
            fd_gen = FractalAnalyzer.calculate_fractal_dimension(sizes_gen, counts_gen)
            fd_gt = FractalAnalyzer.calculate_fractal_dimension(sizes_gt, counts_gt)
            
            batch_loss += F.l1_loss(torch.tensor(fd_gen), torch.tensor(fd_gt))
        
        return batch_loss / len(generated)

    def train(self):
        train_metadata, val_metadata = self.dataset.split_dataset(self.dataset.metadata, val_ratio=0.2)
        
        # Загрузка весов, если требуется
        if self.load_weights:
            try:
                self._load_weights()
                print("Веса моделей загружены успешно")
            except Exception as e:
                print(f"Ошибка загрузки весов: {e}")
        
        # Оптимизаторы
        optimizer_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        
        # Функции потерь
        bce_loss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        
        # Веса для разных компонентов loss
        lambda_l1 = 100.0
        lambda_fractal = 10.0  # Вес для фрактальной компоненты
        
        # Метки для дискриминатора
        real_label = 1.0
        fake_label = 0.0
        
        # История обучения
        g_losses = []
        d_losses = []
        fractal_losses = []
        
        # Подготовка данных валидации заранее
        val_dataset = self.dataset.to_tensor_dataset(val_metadata)
        val_inputs = val_dataset[0][:5].to(self.device)  # Берём только 5 примеров для визуализации
        val_targets = val_dataset[1][:5].to(self.device)
        val_masks = val_dataset[2][:5].to(self.device)
        
        # Основной цикл обучения
        for epoch in range(self.epochs):
            self.model.generator.train()
            self.model.discriminator.train()
            
            # Создаем DataLoader для тренировочных данных
            train_dataset = self.dataset.to_tensor_dataset(train_metadata)
            train_loader = DataLoader(
                torch.utils.data.TensorDataset(train_dataset[0], train_dataset[1], train_dataset[2]),
                batch_size=self.batch_size,
                shuffle=True
            )
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_fractal_loss = 0.0
            
            for batch_idx, (inputs, targets, masks) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                batch_size = inputs.size(0)
                
                # ----------------------
                # Обучение Дискриминатора
                # ----------------------
                optimizer_d.zero_grad()
                
                # Реальные изображения
                real_outputs = self.model.discriminator(targets)
                real_labels = torch.full((batch_size, 1, 1, 1), real_label, device=self.device)
                d_loss_real = bce_loss(real_outputs, real_labels)
                
                # Сгенерированные изображения
                fake_images = self.model.generator(inputs, masks)
                fake_outputs = self.model.discriminator(fake_images.detach())
                fake_labels = torch.full((batch_size, 1, 1, 1), fake_label, device=self.device)
                d_loss_fake = bce_loss(fake_outputs, fake_labels)
                
                # Общая loss дискриминатора
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                d_loss.backward()
                optimizer_d.step()
                
                # ----------------------
                # Обучение Генератора
                # ----------------------
                optimizer_g.zero_grad()
                
                # GAN loss (обмануть дискриминатор)
                fake_outputs_g = self.model.discriminator(fake_images)
                g_loss_gan = bce_loss(fake_outputs_g, real_labels)
                
                # L1 loss (pixel-wise)
                g_loss_l1 = l1_loss(fake_images, targets) * lambda_l1
                
                # Фрактальная loss
                fractal_loss = self._calculate_fractal_loss(targets, fake_images) * lambda_fractal
                
                # Общая loss генератора
                g_loss = g_loss_gan + g_loss_l1 + fractal_loss
                g_loss.backward()
                optimizer_g.step()
                
                # Запись статистики
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_fractal_loss += fractal_loss.item()
                
                # Обновляем прогресс-бар
                pbar.set_postfix({
                    'G_loss': g_loss.item(), 
                    'D_loss': d_loss.item(),
                    'Fractal_loss': fractal_loss.item()
                })
            
            # Средние значения loss за эпоху
            epoch_g_loss /= len(train_loader)
            epoch_d_loss /= len(train_loader)
            epoch_fractal_loss /= len(train_loader)
            
            g_losses.append(epoch_g_loss)
            d_losses.append(epoch_d_loss)
            fractal_losses.append(epoch_fractal_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs} - G_loss: {epoch_g_loss:.4f}, D_loss: {epoch_d_loss:.4f}, Fractal_loss: {epoch_fractal_loss:.4f}")
            
            # Сохранение промежуточных результатов и визуализация
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                self._save_models()
                
                # Визуализация примеров из валидационного набора
                with torch.no_grad():
                    self.model.generator.eval()
                    val_generated = self.model.generator(val_inputs, val_masks)
                    
                    # Визуализация
                    plt.figure(figsize=(15, 10))
                    for i in range(min(5, len(val_inputs))):
                        # Исходное изображение с маской
                        plt.subplot(3, 5, i + 1)
                        plt.imshow(val_inputs[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title('Input')
                        plt.axis('off')
                        
                        # Сгенерированное изображение
                        plt.subplot(3, 5, i + 6)
                        plt.imshow(val_generated[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title('Generated')
                        plt.axis('off')
                        
                        # Целевое изображение
                        plt.subplot(3, 5, i + 11)
                        plt.imshow(val_targets[i].cpu().squeeze().numpy(), cmap='gray')
                        plt.title('Target')
                        plt.axis('off')
                    
                    plt.savefig(os.path.join(self.output_path, f'epoch_{epoch+1}_samples.png'))
                    plt.close()
        
        # Графики обучения
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(g_losses, label='Generator')
        plt.plot(d_losses, label='Discriminator')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Generator and Discriminator Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(fractal_losses, color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Fractal Loss')
        plt.title('Fractal Dimension Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'training_history.png'))
        plt.close()
        
        # Сохранение финальных моделей
        self._save_models()
        
        return g_losses, d_losses, fractal_losses

    def _save_models(self):
        torch.save(self.model.generator.state_dict(), os.path.join(self.output_path, "generator.pth"))
        torch.save(self.model.discriminator.state_dict(), os.path.join(self.output_path, "discriminator.pth"))