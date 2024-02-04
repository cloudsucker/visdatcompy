import numpy as np
from PIL import Image
from utils import get_time
from sklearn.metrics import mean_squared_error as mse_sklearn
from skimage.metrics import normalized_root_mse as nrmse_sklearn


# ==================================================================================================================================
# |                                                              METRICS                                                           |
# ==================================================================================================================================


def pix2pix(first_image_path: str, second_image_path: str) -> bool:
    """
    Функция для сравнения двух изображений методом Pixel to Pixel.

    Вход:
    - first_image_path (string): путь к первому изображению
    - second_image_path (string): путь ко второму изображению

    Вывод:
    - image_equal (boolean): True, если изображения идентичны, иначе False
    """

    resized_image1 = Image.open(first_image_path).resize((512, 512))
    resized_image2 = Image.open(second_image_path).resize((512, 512))

    are_equal = np.array_equal(np.array(resized_image1), np.array(resized_image2))

    return are_equal


# ==================================================================================================================================


@get_time
def mse(first_image_path: str, second_image_path: str) -> float:
    """
    Функция для вычисления среднеквадратичной ошибки (MSE) между двумя изображениями.

    Вход:
    - first_image_path (string): путь к первому изображению
    - second_image_path (string): путь ко второму изображению

    Вывод:
    - result (float): значение MSE между изображениями
    """

    resized_image1 = Image.open(first_image_path).resize((512, 512))
    resized_image2 = Image.open(second_image_path).resize((512, 512))

    result = mse_sklearn(
        np.array(resized_image1).reshape(-1, 2), np.array(resized_image2).reshape(-1, 2)
    )

    return result


# ==================================================================================================================================


def nrmse(first_image_path: str, second_image_path: str) -> bool:
    """
    Функция для вычисления нормализованной среднеквадратичной ошибки (NRMSE) между двумя изображениями.

    Вход:
    - first_image_path (string): путь к первому изображению
    - second_image_path (string): путь ко второму изображению

    Вывод:
    - result (float): значение NRMSE между изображениями
    """

    resized_image1 = np.array(Image.open(first_image_path).resize((512, 512)))
    resized_image2 = np.array(Image.open(second_image_path).resize((512, 512)))

    result = nrmse_sklearn(
        np.array(resized_image1).reshape(-1, 2), np.array(resized_image2).reshape(-1, 2)
    )

    return result


# ==================================================================================================================================

if __name__ == "__main__":
    print(nrmse("dataset/000.jpg", "dataset/001.jpg"))
