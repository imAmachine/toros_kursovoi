import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from generator.gan.gan_inference import ImageGenerator
from generator.shifter.image_shifter import ImageShifter
import os
from settings import ANALYSIS_OUTPUT_FOLDER_PATH

OUTPUT_TEST_INFERENCE_FOLDER_PATH = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'inference')

class ImageShiftApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Shifter App")

        self.image_path = None
        self.mask_path = None
        self.weights_path = None
        self.image = None
        self.mask = None

        self.setup_ui()

    def setup_ui(self):
        # Кнопки загрузки файлов
        tk.Button(self.root, text="Load Weights", command=self.load_weights).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(self.root, text="Load Image", command=self.load_image).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Load Mask", command=self.load_mask).grid(row=0, column=2, padx=5, pady=5)

        # Поля ввода для сдвига по X с направлениями "влево/вправо"
        tk.Label(self.root, text="X Shift (%):").grid(row=1, column=0, padx=5, pady=5)
        self.x_shift_entry = tk.Entry(self.root, width=10)
        self.x_shift_entry.grid(row=1, column=1, padx=5, pady=5)
        self.x_shift_entry.insert(0, "0")  # Значение по умолчанию

        self.x_direction_var = tk.StringVar(value="right")  # Направление по умолчанию вправо
        tk.Radiobutton(self.root, text="Right", variable=self.x_direction_var, value="right").grid(row=1, column=2,
                                                                                                 padx=5, pady=5)
        tk.Radiobutton(self.root, text="left", variable=self.x_direction_var, value="left").grid(row=1, column=3,
                                                                                                   padx=5, pady=5)

        # Поля ввода для сдвига по Y с направлениями "вверх/вниз"
        tk.Label(self.root, text="Y Shift (%):").grid(row=2, column=0, padx=5, pady=5)
        self.y_shift_entry = tk.Entry(self.root, width=10)
        self.y_shift_entry.grid(row=2, column=1, padx=5, pady=5)
        self.y_shift_entry.insert(0, "0")  # Значение по умолчанию

        self.y_direction_var = tk.StringVar(value="up")  # Направление по умолчанию вверх
        tk.Radiobutton(self.root, text="Up", variable=self.y_direction_var, value="up").grid(row=2, column=2, padx=5,
                                                                                             pady=5)
        tk.Radiobutton(self.root, text="Down", variable=self.y_direction_var, value="down").grid(row=2, column=3,
                                                                                                 padx=5, pady=5)

        # Кнопка выполнения
        tk.Button(self.root, text="Apply Shift", command=self.apply_shift).grid(row=3, column=0, columnspan=4, pady=10)

        # Метка для отображения изображения
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=4, column=0, columnspan=4)

    def load_weights(self):
        self.weights_path = filedialog.askopenfilename(filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")])
        if self.weights_path:
            messagebox.showinfo("Info", f"Weights loaded: {self.weights_path}")

    def load_image(self):
        Image.MAX_IMAGE_PIXELS = None
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.tif"), ("All Files", "*.*")])
        if self.image_path:
            self.image = Image.open(self.image_path).convert("L")
            self.display_image(self.image)

    def load_mask(self):
        self.mask_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.tif"), ("All Files", "*.*")])
        if self.mask_path:
            self.mask = Image.open(self.mask_path)
            messagebox.showinfo("Info", "Mask loaded.")

    def apply_shift(self):
        try:
            x_shift_percent = float(self.x_shift_entry.get())
            y_shift_percent = float(self.y_shift_entry.get())

            if abs(x_shift_percent) > 15 or abs(y_shift_percent) > 15:
                messagebox.showerror("Error", "Shift cannot exceed 15%.")
                return

            if not self.image or not self.mask or not self.weights_path:
                messagebox.showerror("Error", "Please load image, mask, and weights.")
                return

            # Учет направления X
            if self.x_direction_var.get() == "left":
                x_shift_percent = -abs(x_shift_percent)
            elif self.x_direction_var.get() == "right":
                x_shift_percent = abs(x_shift_percent)

            # Учет направления Y
            if self.y_direction_var.get() == "down":
                y_shift_percent = -abs(y_shift_percent)
            elif self.y_direction_var.get() == "up":
                y_shift_percent = abs(y_shift_percent)

            # Преобразование в тензоры
            image_tensor = ToTensor()(self.image).unsqueeze(0)
            mask_tensor = ToTensor()(self.mask).unsqueeze(0)

            # Проверка размеров маски и изображения
            if image_tensor.shape != mask_tensor.shape:
                self.image = self.image.resize(self.mask.size)

            # Загрузка модели и выполнение сдвига

            generator = ImageGenerator(generator_weights_path=self.weights_path, image_size=448)
            shifter = ImageShifter(image_size=448)

            os.makedirs(OUTPUT_TEST_INFERENCE_FOLDER_PATH, exist_ok=True)
            transformed_image, transformed_mask, generated_image, generated_mask, img_nodate = generator.generate(self.image,
                                                                                                                  self.mask,
                                                                                                                  x_shift_percent,
                                                                                                                  y_shift_percent,
                                                                                                                  OUTPUT_TEST_INFERENCE_FOLDER_PATH)
            combined_image, combined_mask = shifter.merge_image(transformed_image, transformed_mask,
                                                                generated_image, generated_mask, img_nodate,
                                                                original_sizes=[448, 448], mask_sizes=[448, 448],
                                                                x_shift_percent=x_shift_percent, y_shift_percent=y_shift_percent,
                                                                output_path=OUTPUT_TEST_INFERENCE_FOLDER_PATH)

            combined_image_pil = to_pil_image(combined_image.squeeze(0).cpu())
            self.display_image(combined_image_pil)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_image(self, image):
        image = image.resize((448, 448))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageShiftApp(root)
    root.mainloop()