import numpy as np
from scipy.stats import linregress


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