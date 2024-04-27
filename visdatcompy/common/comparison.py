import pandas as pd

from visdatcompy.common.hash import HASH
from visdatcompy.common.sift import SIFT
from visdatcompy.common.metrics import Metrics
from visdatcompy.common.utils import color_print
from visdatcompy.common.image_handler import Dataset


# ==================================================================================================================================
# |                                                            COMPARISON                                                          |
# ==================================================================================================================================


def compare(dataset1: Dataset, dataset2: Dataset):

    duplicates = []

    # 1. Получение метаданных, сравнение по ним, нахождение полных дублей
    try:
        dataset1.get_exif_data()
        dataset2.get_exif_data()

        color_print("done", "done", "Сравнение по метаданным")

        exif_data = pd.concat(
            [dataset1.exif_data, dataset2.exif_data],
        )

        # Проходимся по дубликатам
        for file_data in exif_data[exif_data.duplicated()].iterrows():
            match_image_name = file_data[1]["Filename"] + file_data[1]["FileExtension"]
            duplicates.append(match_image_name)

            # Убираем дубликаты из объектов
            first_match_index = dataset1.filenames.index(match_image_name)
            dataset1.images.pop(first_match_index)
            dataset1.filenames.pop(first_match_index)
            dataset1.image_count -= 1

            # Убираем дубликаты из объектов
            second_match_index = dataset2.filenames.index(match_image_name)
            dataset2.images.pop(second_match_index)
            dataset2.filenames.pop(second_match_index)
            dataset2.image_count -= 1

        exif_data = exif_data.drop_duplicates()

        if len(duplicates) > 0:
            color_print("done", "done", f"Найдено дупликатов: {len(duplicates)}")

        return duplicates

    except KeyError as e:
        color_print("warning", "warning", "Метаданные не найдены.")

    # 2. Метрики
    # metrics = Metrics(dataset1, dataset2)

    # TODO: Реализовать сравнение по метрикам.

    # 3. ХЭШИ, ORB, FAST.

    # TODO: Реализовать сравнение по Hash и ORB.
    # TODO: Написать FAST.


if __name__ == "__main__":
    # Создаём объекты класса Dataset
    dataset1 = Dataset(r"C:\Users\sharj\Desktop\STUDY\visdatcompy\datasets\drone")
    dataset2 = Dataset(r"C:\Users\sharj\Desktop\STUDY\visdatcompy\datasets\drone")

    # Вызываем функцию для поиска схожестей и дублей
    compare(dataset1, dataset2)
