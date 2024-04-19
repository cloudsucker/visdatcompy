import numpy as np
import pandas as pd
from functools import partial
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error as mse_sklearn
from skimage.metrics import normalized_root_mse as nrmse_skimage
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from sklearn.metrics import mean_absolute_error as mae_skimage
from skimage.metrics import normalized_mutual_information as nmi_skimage

from visdatcompy.common.image_handler import Dataset
from visdatcompy.common.utils import get_time, color_print


__all__ = ["Metrics"]


# ==================================================================================================================================
# |                                                              METRICS                                                           |
# ==================================================================================================================================

# FIXME: Убрать "RuntimeWarning" в common.metrics для метода PSNR.


class Metrics:
    def __init__(self, Dataset1: Dataset, Dataset2: Dataset, results_path: str = ""):
        """
        Класс для сравнения двух датасетов по метрикам.

        Parameters:
            - Dataset1 (object): объект класса Dataset первого датасета.
            - Dataset2 (object): объект класса Dataset второго датасета.
            - results_path: путь для сохранения файлов .csv с результатами.
        """

        self.Dataset1 = Dataset1
        self.Dataset2 = Dataset2

        self.results_path = results_path

    def pix2pix(
        self,
        resize_images: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ) -> list[bool]:

        return self._calculate(np.array_equal, resize_images, to_csv, echo)

    def mae(
        self,
        resize_images: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ) -> list[float]:

        return self._calculate(mae_skimage, resize_images, to_csv, echo)

    def mse(
        self,
        resize_images: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ) -> list[float]:

        return self._calculate(mse_sklearn, resize_images, to_csv, echo)

    def nrmse(
        self,
        resize_images: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ) -> list[float]:

        return self._calculate(nrmse_skimage, resize_images, to_csv, echo)

    def ssim(
        self,
        resize_images: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ) -> list[float]:
        ssim_partial = partial(ssim_skimage, win_size=3)
        ssim_partial.__name__ = "structural_similarity_index"

        return self._calculate(ssim_partial, resize_images, to_csv, echo)

    def psnr(
        self,
        resize_images: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ) -> list[float]:

        return self._calculate(psnr_skimage, resize_images, to_csv, echo)

    def nmi(
        self,
        resize_images: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ) -> list[float]:

        return self._calculate(nmi_skimage, resize_images, to_csv, echo)

    def _calculate(
        self,
        metric_function: object,
        resize_images: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ) -> list[list]:

        result_matrix = []

        with ThreadPoolExecutor() as executor:
            for first_image in self.Dataset1.images:

                row = []

                # Создаем список задач для обработки каждой пары изображений
                futures = []

                for second_image in self.Dataset2.images:
                    if echo:
                        color_print(
                            "log",
                            "log",
                            f"Сравниваем изображения: {first_image.filename} и {second_image.filename}.",
                        )

                    # Субмитим задачу в ThreadPoolExecutor
                    if resize_images:
                        future = executor.submit(
                            metric_function,
                            first_image.read_and_resize(),
                            second_image.read_and_resize(),
                        )
                    else:
                        future = executor.submit(
                            metric_function,
                            first_image.read(),
                            second_image.read(),
                        )

                    # Добавляем объект Future в список
                    futures.append(future)

                # Получаем результаты выполнения задач и добавляем в строку
                for future in futures:
                    row.append(future.result())

                result_matrix.append(row)

        if to_csv == True:
            self.save(metric_function.__name__, result_matrix)

        return result_matrix

    def show(self, metric_values: list[list]) -> None:
        """
        Функция для отображения результата сравнения по метрике в виде тепловой матрицы.

        Parameters:
            - metric_values (list[list]): матрица сравнения по выбранной метрике.
            - filename (string): название файла для сохранения.
        """

        plt.imshow(metric_values, cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.show()

    def save(self, filename: str, metric_values: list[list]) -> None:
        """
        Сохраняет матрицу с результатами сравнения по метрике в csv файл.

        Parameters:
            - filename (string): название файла .csv без расширения.
            - metric_values (list[list]): матрица сравнения по выбранной метрике.
        """
        columns = [image.filename for image in self.Dataset1.images]
        index = [image.filename for image in self.Dataset2.images]

        df = pd.DataFrame(metric_values, columns=columns, index=index)
        df.to_csv(rf"{self.results_path}{filename}.csv")


# ==================================================================================================================================

if __name__ == "__main__":
    dataset1 = Dataset(r"datasets\cows")
    dataset2 = Dataset(r"datasets\cows")

    metrics = Metrics(dataset1, dataset2)

    color_print("log", "log", "Pixel 2 Pixel:")
    pix2pix_result = get_time(metrics.pix2pix)(to_csv=True)
    metrics.show(pix2pix_result)

    color_print("log", "log", "MAE:")
    mae_result = get_time(metrics.mae)()
    metrics.show(mae_result)

    color_print("log", "log", "MSE:")
    mse_result = get_time(metrics.mse)()
    metrics.show(mse_result)

    color_print("log", "log", "NRMSE:")
    nrmse_result = get_time(metrics.nrmse)()
    metrics.show(nrmse_result)

    color_print("log", "log", "SSIM:")
    ssim_result = get_time(metrics.ssim)()
    metrics.show(ssim_result)

    color_print("log", "log", "PSNR:")
    psnr_result = get_time(metrics.psnr)()
    metrics.show(psnr_result)

    color_print("log", "log", "NMI:")
    nmi_result = get_time(metrics.nmi)()
    metrics.show(nmi_result)
