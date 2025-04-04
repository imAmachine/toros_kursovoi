import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
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


class DataAnalyzer:
    """
    Класс для объединения всех компонентов и выполнения анализа.
    """

    def __init__(self, image_dataloader, csv_dataloader):
        self.image_dataloader = image_dataloader
        self.csv_dataloader = csv_dataloader

    def _analyze_images(self):
        """
        Анализ изображений для расчёта фрактальной размерности.
        """
        results = []

        logging.info("Начат анализ изображений.")
        for filename, binary_image in self.image_dataloader.get_all_data():
            sizes, counts = FractalAnalyzer.box_counting(binary_image)
            fractal_dimension = FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
            results.append({
                "filename": filename,
                "fractal_dimension": fractal_dimension
            })

        logging.info("Анализ изображений завершён.")
        return pd.DataFrame(results)

    def extract_coordinates(self, ridge_data):
        """
        Extract coordinate points from the ridge data
        
        Args:
        - ridge_data: DataFrame containing ridge information
        
        Returns:
        - List of coordinate tuples
        """
        # Assuming the coordinate column is stored as a string like "(x, y)"
        coordinates = ridge_data['Value of the first coordinate (X, Y)'].apply(
            lambda x: tuple(map(float, x.strip('()').split(',')))
        ).tolist()
        
        return coordinates

    def analyze(self):
        """
        Основной метод анализа, объединяющий обработку изображений и временных рядов.
        """
        # Анализ изображений
        images_results = self._analyze_images()
        print("\nРезультаты анализа изображений:")
        print(images_results.to_string(index=False, justify='center'))

        # Анализ временных рядов
        # time_series_results = self._analyze_time_series()
        # print("\nРезультаты анализа пространственных рядов:")
        # print(time_series_results.to_string(index=False, justify='center'))

        return images_results#, time_series_results