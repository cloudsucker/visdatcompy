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

from visdatcompy.image_handler import Dataset
from visdatcompy.utils import color_print


__all__ = ["Metrics"]


# ==================================================================================================================================
# |                                                              METRICS                                                           |
# ==================================================================================================================================

# FIXME: Убрать "RuntimeWarning" в common.metrics для метода PSNR.


class Metrics(object):
    """
    Класс для сравнения двух датасетов по метрикам.

    Parameters:
        - Dataset1 (Dataset): объект класса Dataset первого датасета.
        - Dataset2 (Dataset): объект класса Dataset второго датасета.
        - results_path (str): путь для сохранения файлов .csv с результатами.

    Метрики:
    --------
    | Метрика  | Диапазон значений | Описание                                        |
    | -------- | ----------------- | ----------------------------------------------- |
    | PixToPix | True/False        | Попиксельное сравнение двух изображений         |
    | MSE      | [0; ∞)            | Среднеквадратичная ошибка между изображениями   |
    | NRMSE    | [0, 1]            | Нормализованная среднеквадратическая ошибка     |
    | SSIM     | [-1;1]            | Структурное сходство изображений                |
    | PSNR     | (0; ∞)            | Отношение максимального значения сигнала к шуму |
    | MAE      | [0; ∞)            | Средняя абсолютная ошибка между изображениями   |
    | NMI      | [1;2]             | Нормализованный показатель взаимной информации  |

    Методы:
    -------
    - pix2pix(self, resize_images: bool = True, to_csv: bool = False, echo: bool = False) -> list[bool]
      Попиксельное сравнение двух изображений.
    - mse(self, resize_images: bool = True, to_csv: bool = False, echo: bool = False) -> list[float]
      Вычисляет среднеквадратичную ошибку между изображениями.
    - nrmse(self, resize_images: bool = True, to_csv: bool = False, echo: bool = False) -> list[float]
      Вычисляет нормализованную среднеквадратическую ошибку.
    - ssim(self, resize_images: bool = True, to_csv: bool = False, echo: bool = False) -> list[float]
      Вычисляет структурное сходство изображений.
    - psnr(self, resize_images: bool = True, to_csv: bool = False, echo: bool = False) -> list[float]
      Вычисляет отношение максимального значения сигнала к шуму.
    - mae(self, resize_images: bool = True, to_csv: bool = False, echo: bool = False) -> list[float]
      Вычисляет среднюю абсолютную ошибку между изображениями.
    - nmi(self, resize_images: bool = True, to_csv: bool = False, echo: bool = False) -> list[float]
      Вычисляет нормализованный показатель взаимной информации.
    """

    def __init__(self, Dataset1: Dataset, Dataset2: Dataset, results_path: str = ""):
        self.methods = {
            "pix2pix": self.pix2pix,
            "mae": self.mae,
            "mse": self.mse,
            "nrmse": self.nrmse,
            "ssim": self.ssim,
            "psnr": self.psnr,
            "nmi": self.nmi,
        }

        self.Dataset1 = Dataset1
        self.Dataset2 = Dataset2

        self.results_path = results_path

        self.ranges = {
            "mae": {
                "duplicate": self.RangeMask(None, 70),  # 60 -> 70
                "similar": self.RangeMask(70, 119),  # 60 -> 70, 130 -> 119
            },
            "mse": {
                "duplicate": self.RangeMask(None, 60),
                "similar": self.RangeMask(60, 105),  # 110 -> 105
            },
            "nrmse": {
                "duplicate": self.RangeMask(None, 0.2),
                "similar": self.RangeMask(0.2, 0.56),  # 0.6 -> 0.56
            },
            "ssim": {
                "duplicate": self.RangeMask(0.6, 1),
                "similar": self.RangeMask(0.27, 0.6),
            },
            "psnr": {
                "duplicate": self.RangeMask(25, None),
                "similar": self.RangeMask(12, 25),
            },
            "nmi": {
                "duplicate": self.RangeMask(1.3, 2),
                "similar": self.RangeMask(1.007, 1.3),
            },
        }

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

                for second_image in self.Dataset2.images:
                    if echo:
                        color_print(
                            "log",
                            "log",
                            f"Сравниваем изображения: {first_image.filename} и {second_image.filename}.",
                        )

                    # Субмитим задачу в ThreadPoolExecutor
                    if resize_images:
                        value = metric_function(
                            first_image.read_and_resize(),
                            second_image.read_and_resize(),
                        )
                    else:
                        value = metric_function(
                            first_image._read_flatten(),
                            second_image._read_flatten(),
                        )

                    row.append(value)

                result_matrix.append(row)

        if to_csv:
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
        df.to_csv(rf"{self.results_path}/{filename}.csv")

    class RangeMask:
        def __init__(self, lower_bound=None, upper_bound=None):
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        def __call__(self, value):
            if (type(value) == bool) and (value == True):
                return True

            if ((self.lower_bound != None) and (value <= self.lower_bound)) or (
                (self.upper_bound != None) and (value >= self.upper_bound)
            ):
                return False

            return True


# ==================================================================================================================================

if __name__ == "__main__":
    dataset1 = Dataset("datasets/cows")
    dataset2 = Dataset("datasets/cows")

    metrics = Metrics(dataset1, dataset2)

    color_print("log", "log", "Pixel 2 Pixel:")
    pix2pix_result = metrics.pix2pix(echo=True)
    metrics.show(pix2pix_result)

    # color_print("log", "log", "MAE:")
    # mae_result = metrics.mae(echo=True)
    # metrics.show(mae_result)

    # color_print("log", "log", "MSE:")
    # mse_result = metrics.mse(echo=True)
    # metrics.show(mse_result)

    # color_print("log", "log", "NRMSE:")
    # nrmse_result = metrics.nrmse(echo=True)
    # metrics.show(nrmse_result)

    # color_print("log", "log", "SSIM:")
    # ssim_result = metrics.ssim(echo=True)
    # metrics.show(ssim_result)

    # color_print("log", "log", "PSNR:")
    # psnr_result = metrics.psnr(echo=True)
    # metrics.show(psnr_result)

    # color_print("log", "log", "NMI:")
    # nmi_result = metrics.nmi(echo=True)
    # metrics.show(nmi_result)
