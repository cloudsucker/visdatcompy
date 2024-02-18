import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from utils import get_time, color_print, scan_directory
from sklearn.metrics import mean_squared_error as mse_sklearn
from skimage.metrics import normalized_root_mse as nrmse_skimage
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from sklearn.metrics import mean_absolute_error as mae_skimage
from skimage.metrics import normalized_mutual_information as nmi_skimage
from concurrent.futures import ThreadPoolExecutor

from functools import partial

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

        Parameters:
            - image_paths (list[str]): список путей к фотографиям.

        Returns:
            - list[np.array]: список numpy-массивов открытых фотографий.
        """

        with ThreadPoolExecutor() as executor:
            resized_images = list(executor.map(self.load_and_resize_image, image_paths))
            # print(len(resized_images))

        return resized_images

    def load_and_resize_image(self, image_path: str) -> np.array:
        """
        Открывает и масштабирует две фотографии в разрешении 512px x 512px.

        Parameters:
            - image_path (string): путь к фотографии.

        Returns:
            - np.array: numpy массив открытой фотографии.
        """

        with Image.open(image_path) as img:
            img_resized = img.resize((512, 512))
            img_array = np.array(img_resized)
            # print(image_path)

        return img_array.flatten()

    def calculate_metric(
        self, metric_function: object, save_to_csv: bool = False
    ) -> list[float]:
        """
        Функция для сравнения по выбранной метрике

        Parameters:
            - metric_function: объект функции выбранной метрики.

        Returns:
            - list[float]: матрица сравнения по метрике.
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

    def pix2pix(self, save_to_csv=False) -> list[list[bool]]:
        return self.calculate_metric(np.array_equal, save_to_csv)

    def mae(self, save_to_csv=False) -> list[list[float]]:
        return self.calculate_metric(mae_skimage, save_to_csv)

    def mse(self, save_to_csv=False) -> list[list[float]]:
        return self.calculate_metric(mse_sklearn, save_to_csv)

    def nrmse(self, save_to_csv=False) -> list[list[float]]:
        return self.calculate_metric(nrmse_skimage, save_to_csv)

    def ssim(self, save_to_csv=False) -> list[list[float]]:
        # return self.calculate_metric(
        #     lambda im1, im2: ssim_skimage(im1, im2, win_size=3), save_to_csv
        # )
        ssim_partial = partial(ssim_skimage, win_size=3)
        ssim_partial.__name__ = 'ssim_skimage'
        return self.calculate_metric(ssim_partial, save_to_csv)

    def psnr(self, save_to_csv=False) -> list[list[float]]:
        return self.calculate_metric(psnr_skimage, save_to_csv)

    def nmi(self, save_to_csv) -> list[list[float]]:
        return self.calculate_metric(nmi_skimage, save_to_csv)

    def show(self, metric_values: list[list]) -> None:
        """
        Функция для отображения результата сравнения по метрике в виде тепловой матрицы.

        Parameters:
            - metric_values (list[list]): матрица сравнения по выбранной метрике.
        """

        plt.imshow(metric_values, cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.show()

    def save(self, metric_name: str, metric_values: list[list]) -> None:
        """
        Сохраняет матрицу с результатами сравнения по метрике в csv файл.

        Parameters:
            - metric_name (string): название метрики (название создаваемого файла).
            - metric_values (list[list]): матрица сравнения по выбранной метрике.
        """
        df = pd.DataFrame(
            metric_values, columns=self.image_paths2, index=self.image_paths1
        )
        csv_filename = f"metrics_results/{metric_name}.csv"
        df.to_csv(csv_filename)


# ==================================================================================================================================

if __name__ == "__main__":

    # 1. Сканируем директорию нашего датасета и объединяем данные в пути.
    image_data = scan_directory("dataset_small")
    image_paths = list(map(lambda x: os.path.join(x[0], x[1]), image_data))

    # 2. Создаём новый объект класса.
    metrics = Metric(image_paths, image_paths)

    # 3. Сравнение, получение результатов, сохранение в .csv и вывод в виде тепловых матриц:

    # Pixel To Pixel:
    pix2pix_result = get_time(metrics.pix2pix)(True)
    metrics.show(pix2pix_result)

    # MAE:
    mae_result = get_time(metrics.mae)(True)
    metrics.show(mae_result)

    # MSE:
    mse_result = get_time(metrics.mse)(True)
    metrics.show(mse_result)

    # NRMSE:
    nrmse_result = get_time(metrics.nrmse)(True)
    metrics.show(nrmse_result)

    # SSIM:
    ssim_result = get_time(metrics.ssim)(True)  # <=======  Ошибка при создании .csv файла
    metrics.show(ssim_result)

    # PSNR:
    psnr_result = get_time(metrics.psnr)(True)
    metrics.show(psnr_result)

    # NMI:
    nmi_result = get_time(metrics.nmi)(True)
    metrics.show(nmi_result)
