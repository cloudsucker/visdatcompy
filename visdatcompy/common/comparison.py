import os
import pandas as pd

from visdatcompy.common.metrics import Metrics
from visdatcompy.common.utils import scan_directory

metric_list = {
    "pix2pix": Metrics.pix2pix,
    "mse": Metrics.mse,
    "ssim": Metrics.ssim,
    "psnr": Metrics.psnr,
    "mae": Metrics.mae,
}

# ==================================================================================================================================
# |                                                            COMPARISON                                                          |
# ==================================================================================================================================


def compare(
    dataset_path_1: str,
    dataset_path_2: str,
    fast_checking: bool = False,
    metric: str = None,
) -> pd.DataFrame:
    """
    Главная функция для сравнения фотографий.

    Parameters:
        - image_path (str): путь к изображению

    Returns:
        - exif_dict (dict): словарь с метаданными EXIF
    """

    # Сканируем изображения в директории:
    image_data_1 = scan_directory(dataset_path_1)
    image_data_2 = scan_directory(dataset_path_2)

    # Преобразовываем полученные данные в пути:
    image_paths_1 = list(map(lambda x: os.path.join(x[0], x[1]), image_data_1))
    image_paths_2 = list(map(lambda x: os.path.join(x[0], x[1]), image_data_2))

    metrics = Metrics(image_paths_1, image_paths_2)

    if fast_checking == True:
        # TODO: Здесь сделать сравнение по exif-данным.

        pix2pix_result = metrics.pix2pix()
        # TODO: Здесь логика для анализа полученных данных

    # TODO: Реализовать сравнение по выбранным метрикам.
    # TODO: Объединить результаты сравнения метрик в одну тепловую матрицу.


"""
 _________________________________________________________________________________________________________________
|                                                                                                                 |
|                                           ЛОГИКА РАБОТЫ ГЛАВНОЙ ФУНКЦИИ:                                        |
|                                                                                                                 |
| 1. Сравнение по exif-данным.                                                                                    |
| 1.1. С помощью get_exif_data(image_path) получаем exif-данные нашей первой фотки. (ПРЕОБРАЗОВАТЬ В DATAFRAME)   |
| 1.2. С помощью create_exif_dataframe(dataset_path) получаем exif-данные всех изображений. (УЖЕ DATAFRAME)       |
| 1.3. Сравниваем exif-данные фотки с другими изображениями для поиска дублей.                                    |
| 2. Сравнение Pix2Pix.                                                                                           |
| 3. Сравнение по метрикам.                                                                                       |
|                                                                                                                 |
| Returns: таблица схожести.                                                                                      |
|                                                                                                                 |
|_________________________________________________________________________________________________________________|
"""
