import pandas as pd

from visdatcompy.hash import Hash
from visdatcompy.sift import SIFT
from visdatcompy.metrics import Metrics
from visdatcompy.utils import color_print
from visdatcompy.image_handler import Image, Dataset


# ==================================================================================================================================
# |                                                            COMPARISON                                                          |
# ==================================================================================================================================


def find_exif_duplicates(dataset1: Dataset, dataset2: Dataset) -> list[Image]:
    """
    Функция для нахождения полных дублей изображений из первого датасета во втором.

    Parameters:
        - dataset1 (Dataset): Исходный датасет, содержащий оригиналы.
        - dataset2 (Dataset): Датасет, для поиска в нём дублей оригиналов.

    Returns:
        - duplicates (list[Image]): Список с объектами изображений второго датасета,
        являющимися дубликатами изображений первого.

    """

    duplicates = []

    try:
        dataset1.get_exif_data()
        dataset2.get_exif_data()

        color_print("done", "done", "Сравнение по метаданным")

        exif_data = pd.concat(
            [dataset1.exif_data, dataset2.exif_data],
        )

        # Итерация по дубликатам, за исключением имени файла
        for file_data in exif_data[
            exif_data.duplicated(subset=exif_data.columns.difference(["Filename"]))
        ].iterrows():

            # Получаем имя изображения-дубликата
            match_image_name = file_data[1]["Filename"] + file_data[1]["FileExtension"]

            # Убираем дубликаты из объекта датасета, перемещаем их в список дубликатов
            second_match_index = dataset2.filenames.index(match_image_name)
            duplicates.append(dataset2.images[second_match_index])

            dataset2.images.pop(second_match_index)
            dataset2.filenames.pop(second_match_index)
            dataset2.image_count -= 1

        # Удаляем метаданные
        del exif_data

        if len(duplicates) > 0:
            color_print("done", "done", f"Найдено дупликатов: {len(duplicates)}")

        return duplicates

    except AttributeError as e:
        color_print("warning", "warning", f"Метаданные не найдены.")


# 2. Метрики
# metrics = Metrics(dataset1, dataset2)

# TODO: Реализовать сравнение по метрикам.

# 3. ХЭШИ, ORB, FAST.

# TODO: Реализовать сравнение по Hash и ORB.
# TODO: Написать FAST.


if __name__ == "__main__":
    # Создаём объекты класса Dataset
    dataset1 = Dataset("datasets/drone")
    dataset2 = Dataset("datasets/drone_duplicates")

    # Вызываем функцию для поиска схожестей и дублей
    find_exif_duplicates(dataset1, dataset2)
