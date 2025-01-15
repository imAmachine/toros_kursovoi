import matplotlib.pyplot as plt

class InferenceVisualizer:
    @staticmethod
    def visualize(original_image, original_mask, combined_image, combined_mask):
        """
        Визуализация результатов.
        :param original_image: Оригинальное изображение.
        :param original_mask: Оригинальная маска.
        :param combined_image: Комбинированное изображение.
        :param combined_mask: Комбинированная маска.
        """
        plt.figure(figsize=(20, 5))

        # Оригинальное изображение
        plt.subplot(1, 4, 1)
        plt.title("Original Image")
        plt.imshow(original_image.permute(1, 2, 0).clip(0, 1), cmap="gray")
        plt.axis("off")

        # Оригинальная маска
        plt.subplot(1, 4, 2)
        plt.title("Original Mask")
        plt.imshow(original_mask[0].squeeze().clip(0, 1), cmap="gray")
        plt.axis("off")

        # Комбинированное изображение
        plt.subplot(1, 4, 3)
        plt.title("Combined Image")
        plt.imshow(combined_image.clip(0, 1), cmap="gray")
        plt.axis("off")

        # Комбинированная маска
        plt.subplot(1, 4, 4)
        plt.title("Combined Mask")
        plt.imshow(combined_mask.clip(0, 1), cmap="gray")
        plt.axis("off")

        plt.show()