from torchvision.transforms.functional import to_pil_image
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F
from torchvision.transforms import ToTensor

class ImageShifter:
    def __init__(self, image_size):
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = 0.0  # Среднее значение шума
        self.stddev = 0.1  # Стандартное отклонение шума

    def apply_shift(self, image, mask, x_shift_percent, y_shift_percent):
        """
        Создание сдвинутого изображения и маски с заполнением шумом.
        :param image: Исходное изображение (тензор).
        :param mask: Исходная маска (тензор).
        :param x_shift_percent: Сдвиг по оси X (в процентах).
        :param y_shift_percent: Сдвиг по оси Y (в процентах).
        :return: Сдвинутое изображение и маска.
        """
        # Вычисление сдвига в пикселях
        x_shift = int(image.shape[2] * x_shift_percent / 100)
        y_shift = int(image.shape[1] * y_shift_percent / 100)

        # Создание пустых тензоров с шумом
        shifted_image = torch.empty_like(image).normal_(self.mean, self.stddev).to(self.device)
        shifted_mask = torch.empty_like(mask).normal_(self.mean, self.stddev).to(self.device)

        # Вычисление новых границ
        x_start_src, x_end_src = max(0, x_shift), min(image.shape[2], image.shape[2] + x_shift)
        y_start_src, y_end_src = max(0, y_shift), min(image.shape[1], image.shape[1] + y_shift)

        x_start_tgt, x_end_tgt = max(0, -x_shift), min(image.shape[2], image.shape[2] - x_shift)
        y_start_tgt, y_end_tgt = max(0, -y_shift), min(image.shape[1], image.shape[1] - y_shift)

        # Перенос пикселей
        shifted_image[:, y_start_tgt:y_end_tgt, x_start_tgt:x_end_tgt] = \
            image[:, y_start_src:y_end_src, x_start_src:x_end_src]
        shifted_mask[:, y_start_tgt:y_end_tgt, x_start_tgt:x_end_tgt] = \
            mask[:, y_start_src:y_end_src, x_start_src:x_end_src]

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)
        # plt.title("Shifted Image")
        # plt.imshow(shifted_image[0].cpu().numpy(), cmap="gray")

        # plt.subplot(1, 2, 2)
        # plt.title("Shifted Mask")
        # plt.imshow(shifted_mask[0].cpu().numpy(), cmap="gray")

        # plt.show()
        
        # Нормализация и корректировка значений
        shifted_image = torch.clamp(shifted_image, 0, 1)
        shifted_mask = torch.clamp(shifted_mask, 0, 1)

        return shifted_image, shifted_mask

    def _shift(self, x_shift, y_shift, x_shift_percent, y_shift_percent, shape):
        h, w = shape[1], shape[2]  # Размеры изображения

        if x_shift_percent < 0:  # Сдвиг вправо
            x_start_src, x_end_src = 0, w - x_shift
            x_start_tgt, x_end_tgt = x_shift, w
        else:  # Сдвиг влево
            x_start_src, x_end_src = x_shift, w
            x_start_tgt, x_end_tgt = 0, w - x_shift

        if y_shift_percent > 0:  # Сдвиг вниз
            y_start_src, y_end_src = 0, h - y_shift
            y_start_tgt, y_end_tgt = y_shift, h
        else:  # Сдвиг вверх
            y_start_src, y_end_src = y_shift, h
            y_start_tgt, y_end_tgt = 0, h - y_shift

        # Проверяем, чтобы срезы не выходили за пределы и имели ненулевой размер
        x_start_src, x_end_src = max(0, x_start_src), max(0, x_end_src)
        y_start_src, y_end_src = max(0, y_start_src), max(0, y_end_src)
        x_start_tgt, x_end_tgt = max(0, x_start_tgt), max(0, x_end_tgt)
        y_start_tgt, y_end_tgt = max(0, y_start_tgt), max(0, y_end_tgt)

        return x_start_src, x_end_src, x_start_tgt, x_end_tgt, y_start_src, y_end_src, y_start_tgt, y_end_tgt


    def merge_image(self, transformed_image, transformed_mask, generated_image, generated_mask, original_sizes, mask_sizes, x_shift_percent, y_shift_percent, output_path):
        if not isinstance(transformed_image, torch.Tensor):
            transformed_image = ToTensor()(transformed_image).to(self.device)
        if not isinstance(transformed_mask, torch.Tensor):
            transformed_mask = ToTensor()(transformed_mask).to(self.device)
        if not isinstance(generated_image, torch.Tensor):
            generated_image = ToTensor()(generated_image).to(self.device)
        if not isinstance(generated_mask, torch.Tensor):
            generated_mask = ToTensor()(generated_mask).to(self.device)
        
        generated_image_resized = F.interpolate(generated_image, size=(mask_sizes[1], mask_sizes[0]), mode="bilinear").squeeze(0)
        generated_mask_resized = F.interpolate(generated_mask, size=(mask_sizes[1], mask_sizes[0]), mode="bilinear").squeeze(0)
        transformed_image_resized = F.interpolate(transformed_image.unsqueeze(0), size=(mask_sizes[1], mask_sizes[0]), mode="bilinear").squeeze(0)
        transformed_mask_resized = F.interpolate(transformed_mask.unsqueeze(0), size=(mask_sizes[1], mask_sizes[0]), mode="bilinear").squeeze(0)

        x_shift = int(mask_sizes[0] * abs(x_shift_percent) / 100)
        y_shift = int(mask_sizes[1] * abs(y_shift_percent) / 100)

        x_start_src, x_end_src, x_start_tgt, x_end_tgt, y_start_src, y_end_src, y_start_tgt, y_end_tgt = self._shift(x_shift, y_shift, x_shift_percent, y_shift_percent, transformed_image.shape)

        combined_image = torch.zeros((mask_sizes[1] + y_shift, mask_sizes[0] + x_shift), device=self.device)
        combined_mask = torch.zeros((mask_sizes[1] + y_shift, mask_sizes[0] + x_shift), device=self.device)

        combined_image[y_start_src:y_end_src, x_start_src:x_end_src] = generated_image_resized
        combined_mask[y_start_src:y_end_src, x_start_src:x_end_src] = generated_mask_resized

        nodata_resized = (transformed_image_resized != 0).to(self.device)

        combined_image[y_start_tgt:y_end_tgt, x_start_tgt:x_end_tgt] = torch.where(
            nodata_resized,
            transformed_image_resized.to(self.device),
            combined_image[y_start_tgt:y_end_tgt, x_start_tgt:x_end_tgt]
        )

        combined_mask[y_start_tgt:y_end_tgt, x_start_tgt:x_end_tgt] = torch.where(
            nodata_resized,
            transformed_mask_resized.to(self.device),
            combined_mask[y_start_tgt:y_end_tgt, x_start_tgt:x_end_tgt]
        )

        x_shift_orig = int(original_sizes[0] * abs(x_shift_percent) / 100)
        y_shift_orig = int(original_sizes[1] * abs(y_shift_percent) / 100)

        combined_image_resized = F.interpolate(combined_image.unsqueeze(0).unsqueeze(0),
                                               size=(original_sizes[1] + y_shift_orig, original_sizes[0] + x_shift_orig),
                                               mode="bilinear").squeeze()

        # Преобразование в формат PIL
        combined_image_pil = to_pil_image(combined_image_resized.cpu())
        combined_mask_pil = to_pil_image(combined_mask.cpu())

        # Сохранение изображения
        combined_image_pil.save(f"{output_path}/image.tif", format="TIFF")
        combined_mask_pil.save(f"{output_path}/mask.png", format="PNG")

        return combined_image.cpu(), combined_mask.cpu()