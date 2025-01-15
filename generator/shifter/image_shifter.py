from torchvision.transforms.functional import to_pil_image
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

class ImageShifter:
    def __init__(self, image_size):
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply_shift(self, image, mask, x_shift_percent, y_shift_percent):
        """
        Создание сдвинутого изображения и маски с заполнением шумом.
        :param image: Исходное изображение.
        :param mask: Исходная маска.
        :param x_shift_percent: Сдвиг по оси X (в процентах).
        :param y_shift_percent: Сдвиг по оси Y (в процентах).
        :return: Сдвинутое изображение и маска.
        """
        x_shift = int(image.shape[2] * abs(x_shift_percent) / 100)
        y_shift = int(image.shape[1] * abs(y_shift_percent) / 100)

        shifted_image = torch.randn_like(image)  # Белый шум
        shifted_mask = torch.rand_like(mask)

        if x_shift_percent < 0:  # Сдвиг вправо
            x_start_src, x_end_src = 0, -x_shift
            x_start_tgt, x_end_tgt = x_shift, None
        else:  # Сдвиг влево
            x_start_src, x_end_src = x_shift, None
            x_start_tgt, x_end_tgt = 0, -x_shift

        if y_shift_percent > 0:  # Сдвиг вниз
            y_start_src, y_end_src = 0, -y_shift
            y_start_tgt, y_end_tgt = y_shift, None
        else:  # Сдвиг вверх
            y_start_src, y_end_src = y_shift, None
            y_start_tgt, y_end_tgt = 0, -y_shift

        shifted_image[:, y_start_tgt:y_end_tgt, x_start_tgt:x_end_tgt] = image[:, y_start_src:y_end_src, x_start_src:x_end_src]
        shifted_mask[:, y_start_tgt:y_end_tgt, x_start_tgt:x_end_tgt] = mask[:, y_start_src:y_end_src, x_start_src:x_end_src]

        nodata_mask = shifted_image == 0  # Предполагаем, что 0 — это значение nodata
        shifted_image[nodata_mask] = torch.randn_like(shifted_image[nodata_mask])
        shifted_mask[nodata_mask] = torch.rand_like(shifted_mask[nodata_mask])

        shifted_image = transforms.Normalize(mean=[0.5], std=[0.5])(shifted_image)
        return shifted_image, shifted_mask

    def merge_image(self, transformed_image, transformed_mask, generated_image, generated_mask, original_sizes, mask_sizes, x_shift_percent, y_shift_percent, OUTPUT_TEST_INFERENCE_FOLDER_PATH):
        generated_image_resized = F.interpolate(generated_image, size=(mask_sizes[1], mask_sizes[0]),
                                                mode="bilinear").squeeze(0)
        generated_mask_resized = F.interpolate(generated_mask, size=(mask_sizes[1], mask_sizes[0]), mode="bilinear").squeeze(
            0)
        transformed_image_resized = F.interpolate(transformed_image.unsqueeze(0), size=(mask_sizes[1], mask_sizes[0]),
                                                  mode="bilinear").squeeze(0)
        transformed_mask_resized = F.interpolate(transformed_mask.unsqueeze(0), size=(mask_sizes[1], mask_sizes[0]),
                                                 mode="bilinear").squeeze(0)

        x_shift = int(mask_sizes[0] * abs(x_shift_percent) / 100)
        y_shift = int(mask_sizes[1] * abs(y_shift_percent) / 100)

        if x_shift_percent < 0:  # Сдвиг вправо
            x_start_src, x_end_src = 0, -x_shift
            x_start_tgt, x_end_tgt = x_shift, None
        else:  # Сдвиг влево
            x_start_src, x_end_src = x_shift, None
            x_start_tgt, x_end_tgt = 0, -x_shift

        if y_shift_percent > 0:  # Сдвиг вниз
            y_start_src, y_end_src = 0, -y_shift
            y_start_tgt, y_end_tgt = y_shift, None
        else:  # Сдвиг вверх
            y_start_src, y_end_src = y_shift, None
            y_start_tgt, y_end_tgt = 0, -y_shift

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
        combined_image_pil.save(f"{OUTPUT_TEST_INFERENCE_FOLDER_PATH}/image.tif", format="TIFF")
        combined_mask_pil.save(f"{OUTPUT_TEST_INFERENCE_FOLDER_PATH}/mask.png", format="PNG")

        return combined_image.cpu(), combined_mask.cpu()