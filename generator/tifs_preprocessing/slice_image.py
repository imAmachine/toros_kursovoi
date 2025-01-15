import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from osgeo import gdal
from PIL import Image

class ImageMaskSlicer:
    def __init__(self, geo_data_path, image_dir, mask_dir, output_image_dir, output_mask_dir, grid_size=50, target_tiles_count=300):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.geo_data = pd.read_csv(geo_data_path, sep=';')
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir
        self.grid_size = grid_size
        self.target_tiles_count = target_tiles_count

        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_mask_dir, exist_ok=True)

    def calc_tiles_data_by_tif(self, tiff_name):
        """
        Рассчитывает параметры плитки для указанного GeoTIFF файла.
        :param tiff_name: Имя файла GeoTIFF.
        :return: Словарь с рассчитанными параметрами плитки.
        """
        tiff_data = self.geo_data[self.geo_data['file'] == tiff_name]
        assert len(tiff_data) == 1, f"Multiple or no entries found for {tiff_name}"

        img_w = tiff_data['width'].values[0]
        img_h = tiff_data['height'].values[0]

        calc = {
            'tile_area': None,
            'tile_pixels_count': None,
            'tile_resolution': None,
            'stride': None,
            'overlap_coef': None
        }

        calc['tile_area'] = np.float32(tiff_data['ground_area_m2'].values[0]) / self.grid_size
        calc['tile_pixels_count'] = calc['tile_area'] / tiff_data['pixel_area'].values[0]
        calc['tile_resolution'] = int(np.round(np.sqrt(calc['tile_pixels_count']), 0))

        if img_w <= calc['tile_resolution'] or img_h <= calc['tile_resolution']:
            raise ValueError("Tile resolution is too large for the image dimensions.")

        calc['stride'] = int(
            np.round(
                np.sqrt(((img_w - calc['tile_resolution']) * (img_h - calc['tile_resolution'])) / (self.target_tiles_count * 1.1)),
                0
            )
        )

        calc['overlap_coef'] = np.round((calc['stride'] / calc['tile_resolution']) * 100, 2)
        return calc

    def slice_all(self):
        """
        Создает пары изображений и масок на основе имен файлов и разрезает их.
        """
        image_files = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith(".tif")]
        mask_files = [os.path.join(self.mask_dir, f) for f in os.listdir(self.mask_dir) if f.endswith(".png")]

        mask_dict = {os.path.basename(mask).replace("_mask.png", ""): mask for mask in mask_files}

        for image_path in image_files:
            base_name = os.path.basename(image_path).replace("_image.tif", "")

            if base_name not in mask_dict:
                print(f"Warning: No matching mask found for {os.path.basename(image_path)}")
                continue

            mask_path = mask_dict[base_name]

            print(f"Processing {os.path.basename(image_path)} with mask {os.path.basename(mask_path)}")

            try:
                tiff_name = os.path.basename(image_path)
                tile_params = self.calc_tiles_data_by_tif(tiff_name)
                self._slice_image_and_mask(image_path, mask_path, tile_params)
            except Exception as e:
                print(f"Error processing {tiff_name}: {e}")


    def _slice_image_and_mask(self, image_path, mask_path, tile_params):
        """
        Разрезает изображение и маску с учётом параметров плитки и исключает патчи с нулевыми значениями.
        Маска предварительно масштабируется до размера изображения.
        :param image_path: Путь к исходному изображению (TIF).
        :param mask_path: Путь к маске (PNG).
        :param tile_params: Параметры плитки (словарь).
        """
        image_ds = gdal.Open(image_path)

        img_width = image_ds.RasterXSize
        img_height = image_ds.RasterYSize

        tile_size = tile_params['tile_resolution']
        stride = tile_params['stride']

        # Масштабируем маску до размеров изображения
        mask = Image.open(mask_path)
        mask = mask.resize((img_width, img_height), Image.BILINEAR)
        mask_array = np.array(mask)

        total_patches = ((img_width - tile_size) // stride + 1) * ((img_height - tile_size) // stride + 1)
        progress_bar = tqdm(total=total_patches, desc=f"Slicing {os.path.basename(image_path)}")

        saved_patches = 0

        for i in range(0, img_width - tile_size + 1, stride):
            for j in range(0, img_height - tile_size + 1, stride):
                image_patch = self._read_tile(image_ds, i, j, tile_size)
                mask_patch = self._extract_mask_patch(mask_array, i, j, tile_size)

                if self._has_zero_pixels(image_patch):
                    progress_bar.update(1)
                    continue

                base_name = os.path.splitext(os.path.basename(image_path))[0]
                image_patch_name = f"{base_name}_x{i}_y{j}.png"
                mask_patch_name = f"{base_name}_x{i}_y{j}.png"

                self._save_patch(image_patch, os.path.join(self.output_image_dir, image_patch_name))
                self._save_patch(mask_patch, os.path.join(self.output_mask_dir, mask_patch_name))

                saved_patches += 1
                progress_bar.update(1)

        progress_bar.close()
        print(f"Разрезка завершена для: {image_path} и {mask_path}. Сохранено патчей: {saved_patches}")

    @staticmethod
    def _read_tile(dataset, x_offset, y_offset, tile_size):
        """
        Читает плитку из GDAL-объекта.
        :param dataset: GDAL Dataset.
        :param x_offset: Смещение по X.
        :param y_offset: Смещение по Y.
        :param tile_size: Размер плитки.
        :return: NumPy массив плитки.
        """
        band = dataset.GetRasterBand(1)  # Используем только первый канал

        # Определяем размеры окна чтения
        win_xsize = min(tile_size, dataset.RasterXSize - x_offset)
        win_ysize = min(tile_size, dataset.RasterYSize - y_offset)

        # Читаем только нужную область
        tile = band.ReadAsArray(x_offset, y_offset, win_xsize, win_ysize)

        # Если область выходит за границы, дополнить нулями
        if tile.shape != (tile_size, tile_size):
            padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
            padded_tile[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded_tile

        return tile

    @staticmethod
    def _extract_mask_patch(mask_array, x_offset, y_offset, tile_size):
        """
        Извлекает патч из масштабированного массива маски.
        :param mask_array: NumPy массив всей маски.
        :param x_offset: Смещение по X.
        :param y_offset: Смещение по Y.
        :param tile_size: Размер патча.
        :return: NumPy массив патча.
        """
        return mask_array[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]

    @staticmethod
    def _save_patch(patch, output_path):
        """
        Сохраняет патч в файл.
        :param patch: NumPy массив патча.
        :param output_path: Путь для сохранения.
        """
        patch_image = Image.fromarray(patch)
        patch_image.save(output_path, compress_level=1)  # Используем сжатие для PNG

    @staticmethod
    def _has_zero_pixels(patch):
        """
        Проверяет, содержит ли патч хотя бы один нулевой пиксель.
        :param patch: NumPy массив патча.
        :return: True, если в патче есть хотя бы один нулевой пиксель, иначе False.
        """
        return np.any(patch == 0)
