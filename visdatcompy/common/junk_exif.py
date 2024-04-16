import os.path
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

from visdatcompy.common.utils import scan_directory, color_print, get_time


__all__ = ["get_exif_data", "get_exif_from_files", "create_exif_dataframe"]


# ==================================================================================================================================
# |                                                           EXIF HANDLER                                                         |
# ==================================================================================================================================


def get_exif_data(image_path: str) -> dict[str, str]:
    """
    Получение метаданных EXIF изображения.

    Parameters:
        - image_path (str): путь к изображению

    Returns:
        - exif_dict (dict): словарь с метаданными EXIF
    """

    exif_dict = {}

    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()

            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    exif_dict[tag_name] = value
            else:
                color_print("fail", "fail", "EXIF данные не найдены.", True)
                return None
    except FileNotFoundError:
        color_print("fail", "fail", "Указанный файл не найден.", True)
        return None

    return exif_dict


# ==================================================================================================================================


def get_exif_from_files(directory: str) -> list[str]:
    """
    Сбор метаданных EXIF из файлов в указанной директории.

    Parameters:
        - directory (str): путь к директории с изображениями

    Returns:
        - files_exif_list (list): список словарей с метаданными EXIF для каждого изображения
    """

    files_exif_list = []
    file_list = scan_directory(directory)

    for file in file_list:
        exif_data = get_exif_data(os.path.join(file[0], file[1]))

        if exif_data is not None:
            filename, file_extension = os.path.splitext(file[1])
            exif_data.update({"Filename": filename, "FileExtension": file_extension})
            files_exif_list.append(exif_data)

    return files_exif_list


# ==================================================================================================================================


def create_exif_dataframe(dataset_path: str, create_csv: bool) -> pd.DataFrame:
    """
    Создание датафрейма с метаданными EXIF изображений из указанной директории.

    Parameters:
        - dataset_path (str): путь к директории с изображениями.
        - create_csv (bool): флаг для создания csv файла с данными при необходимости.

    Returns:
        - df (pd.DataFrame): датафрейм с метаданными EXIF изображений
    """

    data = get_exif_from_files(dataset_path)
    df = pd.DataFrame(data=data)

    df = df.drop(columns="MakerNote")

    df.to_csv("exif_results/meta.csv", header=True, index=False) if create_csv else None

    return df


# ==================================================================================================================================

if __name__ == "__main__":
    # Тестирование функций с измерением времени выполнения
    directory_path = "dataset"

    get_time(get_exif_data)(os.path.join(directory_path, "000.jpg"))
    get_time(get_exif_from_files)(directory_path)

    df = pd.DataFrame(get_time(create_exif_dataframe)(directory_path, True))
    print(f"\n{df}")
