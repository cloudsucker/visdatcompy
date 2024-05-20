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

        self.duplicates_finder = self.DuplicatesFinder(self.dataset1, self.dataset2)
        self.similars_finder = self.SimilarsFinder(self.dataset1, self.dataset2)

    class DuplicatesFinder(object):
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
            - metrics_duplicates (dict): Словарь дубликатов на основе выбранной метрики, где ключи - оригинальные изображения,
            а значения - списки изображений, являющиеся их дубликатами.
        """

        def __init__(self, dataset1, dataset2):

            self.dataset1: Dataset = dataset1
            self.dataset2: Dataset = dataset2

            self.exif_duplicates: Dict[Image, List[Image]] = {}
            self.metrics_duplicates: Dict[Image, List[Image]] = {}

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

            except AttributeError:
                color_print("warning", "warning", f"Метаданные не найдены.")

        def find_metrics_duplicates(
            self, metric_name: str = "pix2pix"
        ) -> Dict[Image, List[Image]]:
            """
            Функция для нахождения дублей изображений с использованием метрик сравнения.

            Parameters:
                - metric_name (str): Название метрики для поиска дубликатов.

            Returns:
                - dict: Словарь, где ключи - это оригинальные изображения,
                а значения - списки изображений, являющиеся их дубликатами.

            metric_names:
                - pix2pix: Попиксельное сравнение двух изображений.
                - mse: Вычисляет среднеквадратичную ошибку между изображениями.
                - nrmse: Вычисляет нормализованную среднеквадратическую ошибку.
                - ssim: Вычисляет структурное сходство изображений.
                - psnr: Вычисляет отношение максимального значения сигнала к шуму.
                - mae: Вычисляет среднюю абсолютную ошибку между изображениями.
                - nmi: Вычисляет нормализованный показатель взаимной информации.
            """

            metrics = Metrics(self.dataset1, self.dataset2)
            datasets_unique = self.dataset1.path == self.dataset2.path

            metric_func = metrics.methods[metric_name]
            is_duplicate = metrics.ranges[metric_name]["duplicate"]

            result = metric_func(resize_images=False, echo=True)

            duplicates = {}

            for i, row in enumerate(result):
                original_image = self.dataset1.images[i]
                duplicates_for_original = [
                    self.dataset2.images[j]
                    for j, value in enumerate(row)
                    if ((i != j) if datasets_unique else True) and (is_duplicate(value))
                ]
                if duplicates_for_original:
                    duplicates[original_image] = duplicates_for_original

            self.metrics_duplicates = duplicates

            return self.metrics_duplicates

        def clear_duplicates(self):
            for duplicates_list in self.metrics_duplicates.values():
                for duplicate in duplicates_list:
                    self.dataset2.delete_image(duplicate)

            for duplicates_list in self.exif_duplicates.values():
                for duplicate in duplicates_list:
                    self.dataset2.delete_image(duplicate)

        def _exif_equal(self, exif1, exif2):
            ignore_columns = {"Filename", "FileExtension", "DateTimeDigitized"}

            filtered_exif1 = {k: v for k, v in exif1.items() if k not in ignore_columns}
            filtered_exif2 = {k: v for k, v in exif2.items() if k not in ignore_columns}

            return filtered_exif1 == filtered_exif2

    class SimilarsFinder(object):
        """
        Внутренний класс для поиска схожих изображений в двух наборах данных.

        Parameters:
            - dataset1 (Dataset): Первый набор данных изображений.
            - dataset2 (Dataset): Второй набор данных изображений.

        Attributes:
            - dataset1 (Dataset): Первый набор данных изображений.
            - dataset2 (Dataset): Второй набор данных изображений.
            - hash_similars (Dict[Image, Image]): Словарь схожих изображений, найденных с помощью хэширования.
            - features_similars (Dict[Image, List[Image]]): Словарь схожих изображений, найденных с использованием метода извлечения признаков.
            - metrics_similars (Dict[Image, List[Image]]): Словарь схожих изображений, найденных с использованием метрик сравнения.

        """

        def __init__(self, dataset1, dataset2):
            self.dataset1: Dataset = dataset1
            self.dataset2: Dataset = dataset2

            self.hash_similars: Dict[Image, Image] = {}
            self.features_similars: Dict[Image, Image] = {}
            self.metrics_similars: Dict[Image, List[Image]] = {}

        def find_hash_similars(self, method: str = "average") -> Dict[Image, Image]:
            """
            Функция для нахождения пар схожих изображений с помощью хэшей.

            Parameters:
                - method (str): Метод сравнения хэшей.

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

                im1 = self.dataset1.images[self.dataset1.filenames.index(im1_name)]
                im2 = self.dataset2.images[self.dataset2.filenames.index(im2_name)]

                self.hash_similars[im1] = im2

            return self.hash_similars

        def find_features_similars(
            self, extractor_name: str = "sift"
        ) -> Dict[Image, List[Image]]:
            """
            Находит схожие изображения второго датасета для каждого изображения
            первого датасета с использованием выбранного метода извлечения признаков.

            Parameters:
                - extractor_name (str, optional): Метод извлечения признаков для сравнения.
                Возможные значения: "sift", "orb", "fast". По умолчанию "sift".

            Returns:
                - dict: Словарь, в котором ключи - объекты изображений из первого датасета,
                а значения - списки объектов изображений из второго датасета, являющихся схожими.
            """

            fext = FeatureExtractor(self.dataset1, self.dataset2, extractor_name)

            self.features_similars = fext.find_similars()

            return self.features_similars

        def find_metrics_similars(
            self, metric_name: str = "mse"
        ) -> Dict[Image, List[Image]]:
            """
            Находит схожие изображения во втором наборе данных для каждого изображения в первом наборе данных
            с использованием выбранной метрики сравнения.

            Parameters:
                - metric_name (str, optional): Название метрики для сравнения.

            Returns:
                - dict: Словарь, в котором ключи - объекты изображений из первого набора данных,
                а значения - списки объектов изображений из второго набора данных, являющихся схожими.

            metric_names:
                - pix2pix: Попиксельное сравнение двух изображений.
                - mse: Вычисляет среднеквадратичную ошибку между изображениями.
                - nrmse: Вычисляет нормализованную среднеквадратическую ошибку.
                - ssim: Вычисляет структурное сходство изображений.
                - psnr: Вычисляет отношение максимального значения сигнала к шуму.
                - mae: Вычисляет среднюю абсолютную ошибку между изображениями.
                - nmi: Вычисляет нормализованный показатель взаимной информации.
            """

            metrics = Metrics(self.dataset1, self.dataset2)
            datasets_unique = self.dataset1.path == self.dataset2.path

            similars_matrix = metrics.methods[metric_name](
                resize_images=False, echo=True
            )
            is_similar = metrics.ranges[metric_name]["similar"]

            for i, row in enumerate(similars_matrix):
                similar_images = []

                for j, value in enumerate(row):
                    if (i != j if datasets_unique else True) and (is_similar(value)):
                        similar_images.append(self.dataset2.images[j])

                self.metrics_similars[self.dataset1.images[i]] = similar_images

            return self.metrics_similars

        def clear_similars(self):
            for duplicate in self.hash_similars.values():
                self.dataset2.delete_image(duplicate)

            for duplicate in self.features_similars.values():
                self.dataset2.delete_image(duplicate)

            for duplicate_list in self.metrics_similars.values():
                for duplicate in duplicate_list:
                    self.dataset2.delete_image(duplicate)


if __name__ == "__main__":
    # Создаём объекты класса Dataset
    dataset1 = Dataset("datasets/drone")
    dataset2 = Dataset("datasets/drone_duplicates")

    # Создаём объект класса VisDatCompare поиска дублей и схожестей
    compy = VisDatCompare(dataset1, dataset2)

    # Ищем дубликаты изображений методом MSE:
    compy.duplicates_finder.find_metrics_duplicates("mse")
    # Удаляем дубликаты
    compy.duplicates_finder.clear_duplicates()

    # Ищем схожие изображения с помощью SIFT:
    compy.similars_finder.find_features_similars("sift")
    # Удаляем схожие изображения
    compy.similars_finder.clear_similars()
