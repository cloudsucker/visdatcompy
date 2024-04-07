import cv2

from visdatcompy.common.utils import color_print, get_time
from visdatcompy.common.image_handler import Images


def hash_compare(Images1: Images, Images2: Images, compare_method: str = "average"):
    """
    Функция для сравнения двух датасетов с помощью хэшей.

    Parameters:
        - Images1: объект класса Images с первым датасетом (изображением).
        - Images2: объект класса Images со вторым датасетом (изображением).
        - compare_method: метод сравнения.

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

    methods = {
        "average": cv2.img_hash.AverageHash_create(),
        "p": cv2.img_hash.PHash_create(),
        "marr_hildreth": cv2.img_hash.MarrHildrethHash_create(),
        "radial_variance": cv2.img_hash.RadialVarianceHash_create(),
        "block_mean": cv2.img_hash.BlockMeanHash_create(),
        "color_moment": cv2.img_hash.ColorMomentHash_create(),
    }

    try:
        hash_function = methods[compare_method]

        for image1_path in Images1.dataset_paths:
            first_image = cv2.imread(image1_path)

            for image2_path in Images2.dataset_paths:
                second_image = cv2.imread(image2_path)

                first_image_hash = hash_function.compute(first_image)
                second_image_hash = hash_function.compute(second_image)

                result = hash_function.compare(first_image_hash, second_image_hash)

                color_print(
                    "done", "done", f"method: {compare_method}, result: {result}"
                )

    except Exception as e:
        color_print("fail", "fail", f"Ошибка сравнения: {e}")


if __name__ == "__main__":
    Images1 = Images(
        r"C:\Users\sharj\Desktop\STUDY\visdatcompy_datasets\hash_comparing"
    )
    Images2 = Images(
        r"C:\Users\sharj\Desktop\STUDY\visdatcompy_datasets\hash_comparing"
    )

    get_time(hash_compare)(Images1, Images2, "average")
    get_time(hash_compare)(Images1, Images2, "p")
    get_time(hash_compare)(Images1, Images2, "marr_hildreth")
    get_time(hash_compare)(Images1, Images2, "radial_variance")
    get_time(hash_compare)(Images1, Images2, "block_mean")
    get_time(hash_compare)(Images1, Images2, "color_moment")
