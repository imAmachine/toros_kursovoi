import math
import numpy as np
from scipy.stats import linregress
import torch
import torch.nn.functional as F

class FractalAnalyzer:
    """
    Класс для анализа фрактальной размерности.
    """
    @staticmethod
    def box_counting(binary_image):
        """
        Реализация метода box counting.
        """
        sizes = []
        counts = []

        min_side = min(binary_image.shape)
        for size in range(2, min_side // 2 + 1, 2):
            count = 0
            for x in range(0, binary_image.shape[0] + 1, size):
                for y in range(0, binary_image.shape[1] + 1, size):
                    if np.any(binary_image[x:x+size, y:y+size] > 0):
                        count += 1
            
            sizes.append(size)
            counts.append(count)

        return sizes, counts

    @staticmethod
    def calculate_fractal_dimension(sizes, counts, epsilon=1e-10):
        log_sizes = np.log(np.array(sizes))
        log_counts = np.log(np.array(counts) + epsilon)
        slope, intercept, r_value, p_value, std_err = linregress(log_sizes, log_counts)
        return np.abs(slope)


class FractalAnalyzerGPU:
    """
    Класс для анализа фрактальной размерности на GPU с использованием PyTorch.
    Предполагается, что входное бинарное изображение (binary_image) является тензором PyTorch,
    находящимся на GPU, где значения > 0 считаются заполненными.
    """

    @staticmethod
    def box_counting(binary_image):
        """
        Реализация метода box counting на GPU.
        
        Аргументы:
          binary_image: 2D тензор (например, размер [H, W]) с элементами, где > 0 – наличие структуры.  
          
        Возвращает:
          sizes: список размеров боксов.
          counts: список количества боксов, содержащих ненулевые элементы, для соответствующего размера.
        """
        sizes = []
        counts = []
        H, W = binary_image.shape[-2], binary_image.shape[-1]
        min_side = min(H, W)

        # Перебор размеров боксов с шагом 2, от 2 до половины минимальной стороны
        for size in range(2, (min_side // 2) + 1, 2):
            # Чтобы корректно обработать границы, дополнительно падим изображение до кратного размера бокса
            new_H = math.ceil(H / size) * size
            new_W = math.ceil(W / size) * size
            pad_H = new_H - H
            pad_W = new_W - W

            # Паддинг вдоль последних двух осей: (лево, право, верх, низ)
            padded = F.pad(binary_image, (0, pad_W, 0, pad_H), mode="constant", value=0)
            
            # Используем метод unfold, чтобы разбить изображение на неперекрывающиеся окна размера size x size.
            # Полученная форма: (num_boxes_y, num_boxes_x, size, size)
            patches = padded.unfold(0, size, size).unfold(1, size, size)
            
            # Приводим каждое окно к вектору и проверяем, содержит ли оно ненулевые элементы.
            patches_flat = patches.contiguous().view(-1, size * size)
            # Получаем булевский вектор: True, если хотя бы один элемент в патче больше нуля
            patch_has_content = patches_flat.gt(0).any(dim=1)
            count = patch_has_content.sum().item()

            sizes.append(size)
            counts.append(count)

        return sizes, counts

    @staticmethod
    def calculate_fractal_dimension(sizes, counts, epsilon=1e-10, device="cuda"):
        """
        Вычисляет фрактальную размерность на GPU при помощи линейной регрессии.
        
        Аргументы:
          sizes: список размеров боксов (из метода box_counting).
          counts: список количества боксов (из метода box_counting).
          epsilon: маленькое число для предотвращения деления на ноль.
          device: устройство (обычно "cuda"), на котором будут проводиться вычисления.
        
        Возвращает:
          Фрактальная размерность в виде скалярного тензора PyTorch.
        """
        sizes_tensor = torch.tensor(sizes, dtype=torch.float32, device=device)
        counts_tensor = torch.tensor(counts, dtype=torch.float32, device=device)

        log_sizes = torch.log(sizes_tensor)
        log_counts = torch.log(counts_tensor + epsilon)

        # Вычисляем параметры линейной регрессии "вручную"
        x_mean = log_sizes.mean()
        y_mean = log_counts.mean()
        numerator = ((log_sizes - x_mean) * (log_counts - y_mean)).sum()
        denominator = ((log_sizes - x_mean)**2).sum() + epsilon
        slope = numerator / denominator

        return slope.abs()