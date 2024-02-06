import metrics
import pandas as pd
from utils import scan_directory
from exif import get_exif_data

metric_list = {
    "pix2pix": metrics.pix2pix,
    "mse": metrics.mse,
    "ssim": metrics.ssim,
    "psnr": metrics.psnr,
    "mae": metrics.mae,
}

# ==================================================================================================================================
# |                                                            COMPARISON                                                          |
# ==================================================================================================================================


def compare(
    image_path: str,
    dataset_path: str,
    fast_checking: bool = False,
    metric: str = None,
) -> pd.DataFrame:
    """

    ОПИСАНИЕ ПЕРЕДЕЛАТЬ                     <=================================================================

    Получение метаданных EXIF изображения.

    Вход:
    - image_path (str): путь к изображению

    Вывод:
    - exif_dict (dict): словарь с метаданными EXIF
    """

    dataset = scan_directory(dataset_path)
    # Переделать в корректные пути

    current_metric = metric_list[metric]

    # 1. Сравнение по exif-данным.
    # 1.1. С помощью get_exif_data(image_path) получаем exif-данные нашей первой фотки. (ПРЕОБРАЗОВАТЬ В DATAFRAME)
    # 1.2. С помощью create_exif_dataframe(dataset_path) получаем exif-данные всех изображений. (УЖЕ DATAFRAME)
    # 1.3. Сравниваем exif-данные фотки с другими изображениями для поиска дублей.
    # 2. Сравнение Pix2Pix.
    # 3. Сравнение по метрикам.

    # Вывод: таблица схожести.

    for image in dataset:
        if fast_checking:
            current_metric(image_path, image)

    return None


# ==================================================================================================================================
