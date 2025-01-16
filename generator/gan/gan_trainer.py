import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ..data_load.tiff_dataset import TIFDataset
from .gan_arch import GANModel
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os


class GANTrainer:
    def __init__(self, model: GANModel, image_path, mask_path, output_path, epochs, batch_size, lr_g, lr_d,
                 load_weights=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # объявление путей к директориям
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_path = output_path
        
        # параметры обучения
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        
        # объявление модели GAN и загрузка весов
        self.model = model
        self.load_weights = load_weights
        self._init_model_params()
        
        # подготовка датасета и трансформаций
        os.makedirs(self.output_path, exist_ok=True)
        self._prepare_data()

    def _init_model_params(self):
        if self.load_weights: 
            self._load_weights()
        
        # объявление оптимизаторов
        self.optimizer_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr_g)
        self.optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.lr_d)
        
        # объявление loss функций
        self.loss_fn_g = smp.losses.DiceLoss(mode='binary')
        self.loss_fn_d = torch.nn.BCEWithLogitsLoss()

    def _load_weights(self):
        weights_dir = os.path.join(self.output_path, 'weights/')
        os.makedirs(weights_dir, exist_ok=True)
        
        gen_path = os.path.join(weights_dir, 'generator.pth')
        discr_path = os.path.join(weights_dir, 'discriminator.pth')
        if os.path.exists(gen_path) and os.path.exists(discr_path):
            self.model.load_weights(os.path.join(weights_dir, 'generator.pth'),
                                    os.path.join(weights_dir, 'discriminator.pth'))
        else:
            assert 'Ошибка загрузки весов моделей'
    
    def _prepare_data(self):
        dataset = TIFDataset(image_dir=self.image_path, 
                             mask_dir=self.mask_path, 
                             image_transform=self.model.image_transform, 
                             mask_transform=self.model.mask_transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _train_vizualize(self, sample_noised, sample_real, sample_mask):
        noisy_input = sample_noised.cpu()
        real_combined = sample_real.cpu()
        mask = sample_mask.cpu()
        
        _, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(noisy_input[0], cmap='gray')
        axs[0].set_title('Noisy Input (Image + Mask)')
        axs[1].imshow(real_combined[0], cmap='gray')
        axs[1].set_title('Ground Truth Combined')
        axs[2].imshow(mask[0], cmap='gray')
        axs[2].set_title('Mask')
        plt.show()
    
    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            progress_bar = tqdm(self.dataloader, desc=f"Training Epoch {epoch + 1}", leave=False)

            dataset_iter = iter(self.dataloader)
            samples = next(dataset_iter)

            for i in range(4):  # Визуализация первых 4 примеров из батча
                self._train_vizualize(samples[0][i], samples[1][i], samples[2][i])

            for noisy_input, real_combined, original_mask in progress_bar:
                noisy_input = noisy_input.to(self.device)
                real_combined = real_combined.to(self.device)
                original_mask = original_mask.to(self.device)

                # Генерация данных
                generated_combined = self.model.generator(noisy_input)
                generated_mask = generated_combined[:, 1:2, :, :]

                # Обучение дискриминатора
                real_output = self.model.discriminator(real_combined)
                fake_output = self.model.discriminator(generated_combined.detach())

                # Метки с "разглаживанием" (label smoothing)
                real_labels = torch.ones_like(real_output) * 0.9 + \
                              torch.rand_like(real_output) * 0.1
                fake_labels = torch.zeros_like(fake_output) + \
                              torch.rand_like(fake_output) * 0.1

                loss_d_real = self.loss_fn_d(real_output, real_labels)
                loss_d_fake = self.loss_fn_d(fake_output, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) * 0.5

                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

                # Обучение генератора
                fake_output_g = self.model.discriminator(generated_combined)
                loss_g_adv = self.loss_fn_d(fake_output_g, torch.ones_like(fake_output_g))

                # Потери на маску
                loss_g_preserve = F.binary_cross_entropy(
                    generated_mask * original_mask,
                    original_mask
                )

                loss_g = loss_g_adv + 10.0 * loss_g_preserve

                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

                # Обновление прогресс-бара
                progress_bar.set_postfix({
                    "Loss_D": loss_d.item(),
                    "Loss_G": loss_g.item(),
                    "Loss_Preserve": loss_g_preserve.item()
                })
            self._save_models() # сохранение каждую эпоху

        # Сохранение моделей после обучения
        self._save_models()

    def _save_models(self):
        torch.save(self.model.generator.state_dict(), os.path.join(self.output_path, "generator.pth"))
        torch.save(self.model.discriminator.state_dict(), os.path.join(self.output_path, "discriminator.pth"))
