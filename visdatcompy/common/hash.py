import cv2
import pandas as pd

from visdatcompy.common.utils import color_print, get_time
from visdatcompy.common.image_handler import Dataset


# ==================================================================================================================================
# |                                                           HASH HANDLER                                                         |
# ==================================================================================================================================


class HashHandler:
    def __init__(self, Dataset1: Dataset, Dataset2: Dataset, results_path: str = ""):
        """
        Класс для сравнения изображений с помощью хэшей.

        Parameters:
            - Dataset1: объект класса Dataset с первым датасетом.
            - Dataset2: объект класса Dataset со вторым датасетом.
            - results_path: путь для сохранения файлов .csv с результатами.
        """

        self.methods = {
            "average": cv2.img_hash.AverageHash_create(),
            "p": cv2.img_hash.PHash_create(),
            "marr_hildreth": cv2.img_hash.MarrHildrethHash_create(),
            "radial_variance": cv2.img_hash.RadialVarianceHash_create(),
            "block_mean": cv2.img_hash.BlockMeanHash_create(),
            "color_moment": cv2.img_hash.ColorMomentHash_create(),
        }

        self.Dataset1: Dataset = Dataset1
        self.Dataset2: Dataset = Dataset2

        self.results_path = results_path

    def find_similars(
        self,
        compare_method: str = "average",
        return_df: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ):
        """
        Функция для нахождения схожих изображений с помощью хэшей.

        Parameters:
            - compare_method (str): метод сравнения.
            - to_csv (bool): опция экспорта результатов в csv файл.
            - echo (bool): логирование в консоль.

        Returns:
            - pd.DataFrame: пары схожих изображений.

        compare_methods:
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

        results = pd.DataFrame()

        try:
            hash_function = self.methods[compare_method]

            for first_image in self.Dataset1.images:
                min_similarity = float("inf")
                similar_image_name = ""

                for second_image in self.Dataset2.images:
                    if first_image.filename != second_image.filename:
                        first_image_hash = hash_function.compute(first_image.read())
                        second_image_hash = hash_function.compute(second_image.read())

                        if echo:
                            color_print(
                                "log",
                                "log",
                                f"Сравнение [{first_image.filename} - {second_image.filename}]:",
                            )

                        similarity = hash_function.compare(
                            first_image_hash, second_image_hash
                        )

                        if echo:
                            color_print("none", "status", f"Хэш: {similarity}", False)

                        if similarity < min_similarity:
                            min_similarity = similarity
                            similar_image_name = second_image.filename

                        results.loc[first_image.filename, "similar_image_name"] = (
                            similar_image_name
                        )

            if to_csv:
                results.to_csv(
                    f"{self.results_path}{compare_method}.csv", encoding="utf-8"
                )

            if return_df:
                return results
            return True

        except Exception as e:
            color_print("fail", "fail", f"Ошибка сравнения: {e}")

    def hash_matrix(
        self,
        compare_method: str = "average",
        return_df: bool = True,
        to_csv: bool = False,
        echo: bool = False,
    ):
        """
        Функция для построения тепловой матрицы двух датасетов на основе хэшей.

        Parameters:
            - compare_method: метод сравнения.
            - to_csv (bool): опция экспорта результатов в csv файл.
            - echo (bool): логирование в консоль.

        compare_methods:
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

        results = pd.DataFrame()

        try:
            hash_function = self.methods[compare_method]

            for first_image in self.Dataset1.images:
                for second_image in self.Dataset2.images:
                    first_image_hash = hash_function.compute(first_image.read())
                    second_image_hash = hash_function.compute(second_image.read())

                    if echo:
                        color_print(
                            "log",
                            "log",
                            f"Сравнение [{first_image.filename} - {second_image.filename}]:",
                        )

                    similarity = hash_function.compare(
                        first_image_hash, second_image_hash
                    )

                    if echo:
                        color_print("none", "status", f"Хэш: {similarity}", False)

                    if first_image.filename not in results:
                        results[first_image.filename] = {}

                    results.loc[first_image.filename, second_image.filename] = (
                        similarity
                    )

            if to_csv:
                results.to_csv(
                    f"{self.results_path}{compare_method}_matrix.csv", encoding="utf-8"
                )

            if return_df:
                return results
            return True

        except Exception as e:
            color_print("fail", "fail", f"Ошибка сравнения: {e}")


# ==================================================================================================================================


if __name__ == "__main__":
    dataset1 = Dataset(r"C:\Users\sharj\Desktop\STUDY\visdatcompy_datasets\cows")
    dataset2 = Dataset(r"C:\Users\sharj\Desktop\STUDY\visdatcompy_datasets\cows")

    Hashes = HashHandler(dataset1, dataset2)

    color_print("log", "log", "AverageHash:")
    get_time(Hashes.find_similars)("average", to_csv=True)

    color_print("log", "log", "PHash:")
    get_time(Hashes.find_similars)("p", to_csv=True)

    color_print("log", "log", "MarrHildrethHash:")
    get_time(Hashes.find_similars)("marr_hildreth", to_csv=True)

    color_print("log", "log", "RadialVarianceHash:")
    get_time(Hashes.find_similars)("radial_variance", to_csv=True)

    color_print("log", "log", "BlockMeanHash:")
    get_time(Hashes.find_similars)("block_mean", to_csv=True)

    color_print("log", "log", "ColorMomentHash:")
    get_time(Hashes.find_similars)("color_moment", to_csv=True)
