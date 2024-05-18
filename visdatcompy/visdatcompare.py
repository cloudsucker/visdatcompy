from typing import Dict, List

from visdatcompy.hash import Hash
from visdatcompy.metrics import Metrics
from visdatcompy.utils import color_print
from visdatcompy.feature_extractor import FeatureExtractor
from visdatcompy.image_handler import Image, Dataset


__all__ = ["VisDatCompare"]


# ==================================================================================================================================
# |                                                            COMPARISON                                                          |
# ==================================================================================================================================


class VisDatCompare(object):
    """
    Класс для сравнения двух наборов данных изображений и поиска дубликатов.

    Parameters:
        - dataset1 (Dataset): Исходный набор данных.
        - dataset2 (Dataset): Набор данных для сравнения. (Если пути совпадают, используется dataset1).

    Attributes:
        - dataset1 (Dataset): Исходный набор данных.
        - dataset2 (Dataset): Набор данных для сравнения. (Если пути совпадают, используется dataset1).
        - duplicate_finder (DuplicateFinder): Объект класса DuplicateFinder для поиска дубликатов
        на основе EXIF данных и метода Pixel to Pixel.
        - similars_finder (SimilarsFinder): Объект класса SimilarsFinder для поиска схожих изображений.
    """

    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2 if dataset2.path != dataset1.path else dataset1

        self.duplicate_finder = self.DuplicateFinder(self.dataset1, self.dataset2)
        self.similars_finder = self.SimilarsFinder(self.dataset1, self.dataset2)

    class DuplicateFinder(object):
        """
        Внутренний класс для поиска дубликатов изображений на основе EXIF данных и метода Pixel to Pixel.

        Parameters:
            - dataset1 (Dataset): Исходный набор данных.
            - dataset2 (Dataset): Набор данных для сравнения. Если пути совпадают, используется dataset1.

        Attributes:
            - dataset1 (Dataset): Исходный набор данных.
            - dataset2 (Dataset): Набор данных для сравнения.
            - exif_duplicates (dict): Словарь дубликатов на основе EXIF данных, где ключи - изображения из первого набора,
            а значения - списки изображений из второго набора, являющиеся дубликатами.
            - pix2pix_duplicates (dict): Словарь дубликатов на основе метода pixel-to-pixel, где ключи - оригинальные изображения,
            а значения - списки изображений, являющиеся их дубликатами.
        """

        def __init__(self, dataset1, dataset2):

            self.dataset1 = dataset1
            self.dataset2 = dataset2

            self.exif_duplicates: Dict[Image, List[Image]] = {}
            self.pix2pix_duplicates: Dict[Image, List[Image]] = {}

        def find_exif_duplicates(self) -> Dict[Image, List[Image]]:
            """
            Функция для нахождения дублей изображений на основе EXIF данных.

            Returns:
                - dict: Словарь, где ключи - это изображения из первого набора,
                а значения - списки изображений из второго набора, являющиеся дубликатами.
            """

            try:
                self.dataset1.get_exif_data()
                self.dataset2.get_exif_data()

                for i1, exif1 in self.dataset1.exif_data.iterrows():
                    for i2, exif2 in self.dataset2.exif_data.iterrows():
                        if self._exif_equal(exif1, exif2):
                            original_image = self.dataset1.images[i1]
                            match_image = self.dataset2.images[i2]

                            if original_image not in self.exif_duplicates:
                                self.exif_duplicates[original_image] = []

                            self.exif_duplicates[original_image].append(match_image)

                return self.exif_duplicates

            except AttributeError as e:
                color_print("warning", "warning", f"Метаданные не найдены.")

        def find_pix2pix_duplicates(self) -> Dict[Image, List[Image]]:
            """
            Функция для нахождения дублей изображений с использованием метрики pixel-to-pixel сравнения.

            Returns:
                - dict: Словарь, где ключи - это оригинальные изображения,
                а значения - списки изображений, являющиеся их дубликатами.
            """

            metrics = Metrics(self.dataset1, self.dataset2)
            result = metrics.pix2pix(True, echo=True)

            duplicates = {}

            for i, row in enumerate(result):
                original_image = self.dataset1.images[i]
                duplicates_for_original = [
                    self.dataset2.images[j]
                    for j, is_duplicate in enumerate(row)
                    if is_duplicate
                ]
                if duplicates_for_original:
                    duplicates[original_image] = duplicates_for_original

            self.pix2pix_duplicates = duplicates

            return self.pix2pix_duplicates

        def _exif_equal(self, exif1, exif2):
            ignore_columns = {"Filename", "FileExtension", "DateTimeDigitized"}

            filtered_exif1 = {k: v for k, v in exif1.items() if k not in ignore_columns}
            filtered_exif2 = {k: v for k, v in exif2.items() if k not in ignore_columns}

            return filtered_exif1 == filtered_exif2

    class SimilarsFinder(object):
        def __init__(self, dataset1, dataset2):
            self.dataset1 = dataset1
            self.dataset2 = dataset2

            self.hash_similars: Dict[Image, Image] = {}
            self.features_similars: Dict[Image, Image] = {}

        def find_hash_similars(self, method: str = "average") -> Dict[Image, Image]:
            """
            Функция для нахождения пар схожих изображений с помощью хэшей.

            Parameters:
                - method (str): метод сравнения.

            Returns:
                - dict: Словарь, где ключи - это оригинальные изображения,
                а значения - изображение, являющееся их дубликатами.

            methods:
                - "average": Рассчитывает хэш-значение на основе среднего значения пикселей,
                быстрый алгоритм хэширования изображений, но подходит только для простых случаев.
                - "p": Улучшенная версия AverageHash, которая медленнее, чем AverageHash, но может
                адаптироваться к более широкому спектру ситуаций.
                - "marr_hildreth": Значение хэша рассчитывается на основе оператора граней
                Марра-Хилдрета, что является самым медленным, но более дискриминативным методом.
                - "radial_variance": Рассчитывает хэш-значение на основе преобразования Радона.
                - "block_mean": Рассчитывает хэш-значение на основе среднего значения блоков,
                представленного в том же статье, что и MarrHildrethHash.
                - "color_moment": Рассчитывает хэш-значение на основе моментов цвета,
                представленного в той же статье, что и RadialVarianceHash.
            """

            hash = Hash(self.dataset1, self.dataset2)
            hash_similars_df = hash.find_similars(method, echo=True)

            if len(self.hash_similars.keys()) > 0:
                del self.hash_similars

            for im1_name, im2_row in hash_similars_df.iterrows():
                im2_name = im2_row["similar_image_name"]

                im1 = dataset1.images[dataset1.filenames.index(im1_name)]
                im2 = dataset2.images[dataset2.filenames.index(im2_name)]

                self.hash_similars[im1] = im2

            return self.hash_similars

        def find_features_similars(
            self, extractor_name: str = "sift"
        ) -> Dict[Image, List[Image]]:
            """
            Находит схожие изображения второго датасета для каждого изображения первого датасета
            с использованием выбранного метода извлечения признаков.

            Args:
                - extractor_name (str, optional): Метод извлечения признаков для сравнения.
                Доступные значения: "sift", "orb", "fast". По умолчанию "sift".

            Returns:
                - dict: Словарь, в котором ключами являются объекты изображений из первого датасета,
                а значениями - списки объектов изображений из второго датасета, являющихся схожими.
            """

            fext = FeatureExtractor(self.dataset1, self.dataset2, extractor_name)

            self.features_similars = fext.find_similars()

            return self.features_similars


if __name__ == "__main__":
    # Создаём объекты класса Dataset
    dataset1 = Dataset("datasets/drone")
    dataset2 = Dataset("datasets/drone_duplicates")

    # Создаём объект класса DatasetsCompare поиска дублей и схожестей
    compy = VisDatCompare(dataset1, dataset2)

    # Ищем дубликаты изображений по метаданным:
    compy.duplicate_finder.find_exif_duplicates()
    print(compy.duplicate_finder.exif_duplicates)

    # Ищем схожие изображения с помощью дескрипторов:
    compy.similars_finder.find_features_similars("sift")
    print(compy.similars_finder.features_similars)

    # Визуализируем полученные с помощью дескрипторов пары изображений:
    for im1, im2 in compy.similars_finder.features_similars.items():
        im1.visualize()
        im2.visualize()
