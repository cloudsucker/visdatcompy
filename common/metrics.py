import numpy as np
from PIL import Image
from utils import get_time
from sklearn.metrics import mean_squared_error as mse_sklearn
from skimage.metrics import normalized_root_mse as nrmse_skimage
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from sklearn.metrics import mean_absolute_error as mae_skimage
from skimage.metrics import normalized_mutual_information as nmi_skimage


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


def nrmse(first_image_path: str, second_image_path: str) -> float:
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

    result = nrmse_skimage(
        np.array(resized_image1).reshape(-1, 2), np.array(resized_image2).reshape(-1, 2)
    )

    return result


# ==================================================================================================================================


def ssim(first_image_path: str, second_image_path: str) -> bool:
    """
    Функция для вычисления структурного сходства (SSIM) между двумя изображениями.

    Вход:
    - first_image_path (string): путь к первому изображению
    - second_image_path (string): путь ко второму изображению

    Вывод:
    - result (float): значение SSIM между изображениями
    """

    resized_image1 = np.array(Image.open(first_image_path).resize((512, 512)))
    resized_image2 = np.array(Image.open(second_image_path).resize((512, 512)))

    result = ssim_skimage(resized_image1, resized_image2, win_size=3)

    return result


# ==================================================================================================================================


def psnr(first_image_path: str, second_image_path: str) -> float:
    """
    Функция для вычисления отношения сигнал/шум (PSNR) между двумя изображениями.

    Вход:
    - first_image_path (string): путь к первому изображению
    - second_image_path (string): путь ко второму изображению

    Вывод:
    - result (float): значение PSNR между изображениями
    """

    resized_image1 = np.array(Image.open(first_image_path).resize((512, 512)))
    resized_image2 = np.array(Image.open(second_image_path).resize((512, 512)))

    result = psnr_skimage(resized_image1, resized_image2)

    return result


# ==================================================================================================================================


def mae(first_image_path: str, second_image_path: str) -> float:
    """
    Функция для вычисления средней абсолютной ошибки (MAE) между двумя изображениями.

    Вход:
    - first_image_path (string): путь к первому изображению
    - second_image_path (string): путь ко второму изображению

    Вывод:
    - result (float): значение MAE между изображениями
    """

    resized_image1 = np.array(Image.open(first_image_path).resize((512, 512)))
    resized_image2 = np.array(Image.open(second_image_path).resize((512, 512)))

    result = mae_skimage(resized_image1.flatten(), resized_image2.flatten())

    return result


# ==================================================================================================================================


def nmi(first_image_path: str, second_image_path: str) -> float:
    """
    Функция для вычисления нормализованной взаимной информации (NMI) между двумя изображениями.

    Вход:
    - first_image_path (string): путь к первому изображению
    - second_image_path (string): путь ко второму изображению

    Вывод:
    - result (float): значение NMI между изображениями
    """

    resized_image1 = np.array(Image.open(first_image_path).resize((512, 512)))
    resized_image2 = np.array(Image.open(second_image_path).resize((512, 512)))

    result = nmi_skimage(resized_image1.flatten(), resized_image2.flatten())

    return result


# ==================================================================================================================================

if __name__ == "__main__":
    # Тест всех функций с измерением времени выполнения.
    image_path1 = "dataset/000.jpg"
    image_path2 = "dataset/001.jpg"

    # Pixel to Pixel Comparison
    result_pix2pix = get_time(pix2pix)(image_path1, image_path2)
    print(f"Pixel to Pixel Comparison: {result_pix2pix}")

    # MSE
    result_mse = get_time(mse)(image_path1, image_path2)
    print(f"MSE: {result_mse}")

    # NRMSE
    result_nrmse = get_time(nrmse)(image_path1, image_path2)
    print(f"NRMSE: {result_nrmse}")

    # SSIM
    result_ssim = get_time(ssim)(image_path1, image_path2)
    print(f"SSIM: {result_ssim}")

    # PSNR
    result_psnr = get_time(psnr)(image_path1, image_path2)
    print(f"PSNR: {result_psnr}")

    # MAE
    result_mae = get_time(mae)(image_path1, image_path2)
    print(f"MAE: {result_mae}")

    # NMI
    result_nmi = get_time(nmi)(image_path1, image_path2)
    print(f"NMI: {result_nmi}")
