import os
import cv2
import pandas as pd

from visdatcompy.common.utils import color_print, get_time
from visdatcompy.common.image_handler import Images


class HashHandler:
    def __init__(
        self,
        Images1: Images,
        Images2: Images,
        results_path: str = "results/hash_results/",
    ):
        """
        Класс для сравнения изображений с помощью хэшей.

        Parameters:
            - Images1: объект класса Images с первым датасетом (изображением).
            - Images2: объект класса Images со вторым датасетом (изображением).
        """

        self.methods = {
            "average": cv2.img_hash.AverageHash_create(),
            "p": cv2.img_hash.PHash_create(),
            "marr_hildreth": cv2.img_hash.MarrHildrethHash_create(),
            "radial_variance": cv2.img_hash.RadialVarianceHash_create(),
            "block_mean": cv2.img_hash.BlockMeanHash_create(),
            "color_moment": cv2.img_hash.ColorMomentHash_create(),
        }

        self.Images1 = Images1
        self.Images2 = Images2

        self.results_path = results_path

    @get_time
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

            for image1_path in self.Images1.dataset_paths:
                first_image = cv2.imread(image1_path)
                first_image_name = os.path.basename(image1_path)

                min_similarity = float("inf")
                similar_image_name = ""

                for image2_path in self.Images2.dataset_paths:
                    second_image = cv2.imread(image2_path)
                    second_image_name = os.path.basename(image2_path)

                    if first_image_name != second_image_name:
                        first_image_hash = hash_function.compute(first_image)
                        second_image_hash = hash_function.compute(second_image)

                        if echo:
                            color_print(
                                "log",
                                "log",
                                f"Сравнение [{first_image_name} - {second_image_name}]:",
                            )

                        similarity = hash_function.compare(
                            first_image_hash, second_image_hash
                        )

                        if echo:
                            color_print("none", "status", f"Хэш: {similarity}", False)

                        if similarity < min_similarity:
                            min_similarity = similarity
                            similar_image_name = second_image_name

                        results.loc[first_image_name, "similar_image_name"] = (
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

            for image1_path in self.Images1.dataset_paths:
                first_image = cv2.imread(image1_path)
                first_image_name = os.path.basename(image1_path)

                for image2_path in self.Images2.dataset_paths:
                    second_image = cv2.imread(image2_path)
                    second_image_name = os.path.basename(image2_path)

                    first_image_hash = hash_function.compute(first_image)
                    second_image_hash = hash_function.compute(second_image)

                    if echo:
                        color_print(
                            "log",
                            "log",
                            f"Сравнение [{first_image_name} - {second_image_name}]:",
                        )

                    similarity = hash_function.compare(
                        first_image_hash, second_image_hash
                    )

                    if echo:
                        color_print("none", "status", f"Хэш: {similarity}", False)

                    if first_image_name not in results:
                        results[first_image_name] = {}

                    results.loc[first_image_name, second_image_name] = similarity

            if to_csv:
                results.to_csv(
                    f"{self.results_path}{compare_method}_matrix.csv", encoding="utf-8"
                )

            if return_df:
                return results
            return True

        except Exception as e:
            color_print("fail", "fail", f"Ошибка сравнения: {e}")


if __name__ == "__main__":
    Images1 = Images(r"C:\Users\sharj\Desktop\STUDY\visdatcompy_datasets\cows")
    Images2 = Images(r"C:\Users\sharj\Desktop\STUDY\visdatcompy_datasets\cows")

    Hashes = HashHandler(Images1, Images2)

    color_print("status", "status", "AverageHash:")
    Hashes.find_similars("average", to_csv=True)

    color_print("status", "status", "PHash:")
    Hashes.find_similars("p", to_csv=True)

    color_print("status", "status", "MarrHildrethHash:")
    Hashes.find_similars("marr_hildreth", to_csv=True)

    color_print("status", "status", "RadialVarianceHash:")
    Hashes.find_similars("radial_variance", to_csv=True)

    color_print("status", "status", "BlockMeanHash:")
    Hashes.find_similars("block_mean", to_csv=True)

    color_print("status", "status", "ColorMomentHash:")
    Hashes.find_similars("color_moment", to_csv=True)

    # similars = pd.read_csv("average.csv", encoding="utf-8")

    # default_path = r"C:\Users\sharj\Desktop\STUDY\visdatcompy_datasets\cows"
    # Img = Images(default_path)

    # for _, row in similars.iterrows():
    #     Img.visualize(row[0])
    #     Img.visualize(row["similar_image_name"])
