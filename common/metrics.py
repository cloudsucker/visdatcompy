from matplotlib import pyplot as plt
import numpy as np
from numpy import array_equal
from PIL import Image
import pandas as pd
from utils import get_time, color_print
from sklearn.metrics import mean_squared_error as mse_sklearn
from skimage.metrics import normalized_root_mse as nrmse_skimage
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from sklearn.metrics import mean_absolute_error as mae_skimage
from skimage.metrics import normalized_mutual_information as nmi_skimage
from concurrent.futures import ThreadPoolExecutor


# ==================================================================================================================================
# |                                                              METRICS                                                           |
# ==================================================================================================================================

class Metric:
    def __init__(self, image_paths1: list, image_paths2: list):

        # Передаём списки путей в локальные переменные:
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2

        # Закидываем уже открытые фотки в списки массивов:
        self.resized_images1 = self.load_and_resize_images(image_paths1)
        self.resized_images2 = self.load_and_resize_images(image_paths2)

    def load_and_resize_images(self, image_paths: list[str]) -> list[np.array]:
        """
        Открывает и масштабирует список фотографий в разрешении 512px x 512px
        с использованием многопоточности.

        Вход:
            - image_paths (list[str]): список путей к фотографиям.

        Вывод:
            - list[np.array]: список numpy-массивов открытых фотографий.
        """

        with ThreadPoolExecutor() as executor:
            resized_images = list(executor.map(self.load_and_resize_image, image_paths))
            # print(len(resized_images))

        return resized_images

    def load_and_resize_image(self, image_path: str) -> np.array:
        """
        Открывает и масштабирует две фотографии в разрешении 512px x 512px.

        Вход:
            - image_path (string): путь к фотографии.

        Вывод:
            - np.array: numpy массив открытой фотографии.
        """

        with Image.open(image_path) as img:
            img_resized = img.resize((512, 512))
            img_array = np.array(img_resized)
            # print(image_path)

        return img_array.flatten()

    def calculate_metric(self, metric_function: object, save_to_csv) -> list[float]:
        """
        Функция для сравнения по выбранной метрике

        Вход:
            - metric_function: объект функции выбранной метрики.
        Вывод:
            -                                      <==========================        WTF IS THIS
        """

        metric_values = []

        with ThreadPoolExecutor() as executor:
            for img1 in self.resized_images1:
                row = list(
                    executor.map(
                        lambda img2: metric_function(img1, img2), self.resized_images2
                    )
                )
                metric_values.append(row)
        if save_to_csv == True:
            self.save(metric_function.__name__, metric_values)
        return metric_values

    def pix2pix(self, save_to_csv=False) -> list:
        return self.calculate_metric(array_equal, save_to_csv)

    def mae(self, save_to_csv=False) -> list:
        return self.calculate_metric(mae_skimage, save_to_csv)

    def mse(self, save_to_csv=False) -> list:
        return self.calculate_metric(mse_sklearn, save_to_csv)

    def nrmse(self, save_to_csv=False) -> list:
        return self.calculate_metric(nrmse_skimage, save_to_csv)

    def ssim(self, save_to_csv=False) -> list:
        return self.calculate_metric(
            lambda x, y: ssim_skimage(x, y, win_size=3), save_to_csv
        )  # < ================   WTF IS THIS

    def psnr(self, save_to_csv=False) -> list:
        return self.calculate_metric(psnr_skimage, save_to_csv)

    def nmi(self, save_to_csv) -> list:
        return self.calculate_metric(nmi_skimage, save_to_csv)

    def show(self, matrix):
        plt.imshow(matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.show()
    def save(self, metric_name: str, metric_values):
        df = pd.DataFrame(metric_values, columns = self.image_paths2, index = self.image_paths1)
        csv_filename = f'{metric_name}.csv'
        df.to_csv(csv_filename)
        print(f"CSV файл '{csv_filename}' успешно создан.")


# ==================================================================================================================================

if __name__ == "__main__":
    print("starting test...")

    #image_paths1 = ["test_images/PSNR-base.jpg", "test_images/PSNR-90.jpg", "test_images/PSNR-30.jpg", "test_images/PSNR-10.jpg"]
    #image_paths2 = ["test_images/PSNR-base.jpg", "test_images/PSNR-90.jpg", "test_images/PSNR-30.jpg", "test_images/PSNR-10.jpg"]
    
    from utils import scan_directory
    import os

    image_paths3 = scan_directory("dataset")
    x2 = list(map(lambda x: os.path.join(x[0], x[1]), image_paths3))
    print()

    metric = Metric(x2, x2)

    #pix2pix_values = metric.pix2pix(True)
    mae_values = metric.mae(True)
    # mse_values = metric.mse()
    # nrmse_values = metric.nrmse()
    # ssim_values = metric.ssim()
    # psnr_values = metric.psnr()
    # nmi_values = metric.nmi()

    #print(f"pix2pix values: {pix2pix_values}")
    print(f"mae values: {mae_values}")
    # print(f"MSE values: {mse_values}")
    # print(f"NRMSE values: {nrmse_values}")
    # print(f"SSIM values: {ssim_values}")
    # print(f"PSNR values: {psnr_values}")
    # print(f"NMI values: {nmi_values}")

    # metric.show(pix2pix_values)
    metric.show(mae_values)
    # metric.show(mse_values)
    # metric.show(nrmse_values)
    # metric.show(ssim_values)
    # metric.show(psnr_values)
    # metric.show(nmi_values)
