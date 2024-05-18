import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict

from visdatcompy.utils import color_print
from visdatcompy.image_handler import Image, Dataset


__all__ = ["FeatureExtractor"]


# ==================================================================================================================================
# |                                                         FEATURE EXTRACTOR                                                      |
# ==================================================================================================================================


class FeatureExtractor(object):
    def __init__(self, dataset1: Dataset, dataset2: Dataset, extractor: str = "sift"):
        """
        Класс для поиска схожих изображений с помощью SIFT, ORB и FAST.

        Attributes:
            - dataset1 (Dataset): Объект класса Dataset.
            - dataset2 (Dataset): Объект класса Dataset.
            - extractor (string): метод сравнения (sift, orb или fast).

        Methods:
            - extract_features(dataset: Dataset): Извлекает дескрипторы из датасета
            и помещает их в атрибуты датасета.
            - find_similar_image(target_image: Image, dataset: Dataset): ищет схожее
            изображение в датасете.
            - visualize_similar_images(target_image: Image, dataset: Dataset): ищет
            схожие изображения и визуализирует найденную пару.
        """

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        self.extractor_names = {
            "sift": "sift",
            "orb": "orb",
            "fast": "fast",
        }

        self.extractors = {
            "sift": cv2.SIFT_create(),
            "orb": cv2.ORB_create(),
            "fast": cv2.FastFeatureDetector_create(),
        }

        self.desc_arr_shapes = {
            "sift": (10000000, 128),
            "orb": (10000000, 32),
            "fast": (10000000, 32),
        }

        self.desc_arr_shape = self.desc_arr_shapes[extractor]

        self.extractor_name = extractor
        self.extractor = self.extractors[extractor]

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        self.dataset1 = dataset1

        color_print("done", "done", "Извлечение дескрипторов")
        color_print("status", "status", f"Датасет 1: {dataset1.name}")
        self._extract_features_from_dataset(dataset1)

        if dataset2.path != dataset1.path:
            self.dataset2 = dataset2
            print("\n")
            color_print("status", "status", f"Датасет 2: {dataset2.name}")
            self._extract_features_from_dataset(dataset2)

        else:
            self.dataset2 = dataset1
            color_print("status", "status", f"Второй датасет дублирует первый.")
            print("\n")

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        if self.extractor_name == "sift":
            self.sift_similars: Dict[Image, Image] = {}
        elif self.extractor_name == "orb":
            self.orb_similars: Dict[Image, Image] = {}
        elif self.extractor_name == "fast":
            self.fast_similars: Dict[Image, Image] = {}

    def find_similars(self) -> Dict[Image, Image]:
        """
        Находит схожие изображения для каждого изображения из первого датасета во втором датасете
        и записывает их в словарь.

        Returns:
            - dict: Словарь, где ключи - объекты изображений из первого датасета,
            а значения - объекты изображений из второго датасета, являющиеся схожими.
        """

        similars_dict: Dict[Image, Image] = {}

        for image in self.dataset1.images:
            similar_image = self._find_similar_image(image, self.dataset2)
            similars_dict[image] = similar_image

        setattr(self, self.extractor_name + "_similars", similars_dict)

        return similars_dict

    def _extract_features_from_image(self, image: Image) -> np.ndarray:
        img = image._read_image_as_rgb()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp = self.extractor.detect(img_gray, None)

        if self.extractor_name != "fast":
            kp, descriptors = self.extractor.compute(img_gray, kp)
        else:
            kp, descriptors = self.extractors["orb"].compute(img_gray, kp)

        if descriptors is None:
            return np.array(
                []
            )  # Возвращает пустой массив, если не удается получить дескрипторы

        return descriptors

    def _extract_features_from_dataset(
        self, dataset: Dataset, echo: bool = True
    ) -> tuple:
        # Инициализация переменных для хранения дескрипторов и меток
        descriptor_count = 0
        descriptors_array = np.zeros(self.desc_arr_shape)
        labels_array = np.zeros((10000000,))

        for i, image in enumerate(dataset.images):

            img = image._read_image_as_rgb()
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            kp = self.extractor.detect(img_gray, None)

            if self.extractor_name != "fast":
                kp, descriptors = self.extractor.compute(img_gray, kp)
            else:
                kp, descriptors = self.extractors["orb"].compute(img_gray, kp)

            current_descriptors_count = descriptors.shape[0]

            for desc in range(current_descriptors_count):
                current_descriptor = descriptors[desc, :]

                # Нормализация дескриптора и запись его в массив descriptors_array
                descriptors_array[descriptor_count, :] = (
                    current_descriptor / np.linalg.norm(current_descriptor)
                )

                # Присвоение метки i для текущего дескриптора в массиве labels_array
                labels_array[descriptor_count] = i
                descriptor_count += 1

        # Обрезка массивов дескрипторов и меток до фактического размера
        extracted_descriptors = descriptors_array[0:descriptor_count, :]
        extracted_labels = labels_array[0:descriptor_count]

        if echo:
            color_print(
                "done",
                "done",
                f"Количество дескрипторов в {str(dataset.image_count)} изображениях: {str(descriptor_count)}",
            )

        setattr(dataset, self.extractor_name + "_descriptors", extracted_descriptors)
        setattr(dataset, self.extractor_name + "_descriptors_labels", extracted_labels)

    def _find_similar_image(self, target_image: Image, dataset: Dataset) -> Image:
        descriptors = getattr(dataset, self.extractor_name + "_descriptors")
        descriptors_labels = getattr(
            dataset, self.extractor_name + "_descriptors_labels"
        )

        try:
            # Попробуем найти индекс целевого изображения в датасете
            target_image_index = dataset.images.index(target_image)

            # Индексы дескрипторов из того же изображения
            same_class_indices = np.where(descriptors_labels == target_image_index)[0]

            # Индексы дескрипторов из других изображений
            different_class_indices = np.where(
                descriptors_labels != target_image_index
            )[0]

            # Дескрипторы целевого изображения
            target_descriptors = descriptors[same_class_indices, :]

        except ValueError:
            # Если целевого изображения нет в датасете
            target_descriptors = self._extract_features_from_image(target_image)

            if target_descriptors is None or target_descriptors.size == 0:
                color_print(
                    "fail",
                    "fail",
                    "Не удалось извлечь дескрипторы из целевого изображения.",
                )
                return False

            different_class_indices = np.arange(len(descriptors_labels))

        # Дескрипторы других изображений
        other_descriptors = descriptors[different_class_indices, :]
        other_labels = descriptors_labels[different_class_indices]

        # Вычисление попарного скалярного произведения между дескрипторами целевого и других изображений
        dot_products = np.dot(other_descriptors, target_descriptors.T)

        # Количество дескрипторов целевого изображения
        num_test_descriptors = target_descriptors.shape[0]

        # Массивы для хранения максимальных значений скалярного произведения и меток изображений
        max_dot_products = np.zeros((num_test_descriptors,))
        corresponding_labels = []

        # Выбор наиболее похожего изображения на основе скалярного произведения
        for k in range(num_test_descriptors):
            dot_product_values = dot_products[:, k]
            max_value = dot_product_values.max()
            max_dot_products[k] = max_value
            max_index = np.where(dot_product_values == max_value)
            corresponding_labels.extend(other_labels[max_index[0]])

        # Преобразование corresponding_labels в одномерный массив
        corresponding_labels = np.array(corresponding_labels, dtype=int)

        # Определение наиболее часто встречающейся метки среди изображений с высокой похожестью
        high_similarity_indices = np.where(max_dot_products > 0.9)[0]

        if high_similarity_indices.size > 0:
            most_common_label = stats.mode(
                corresponding_labels[high_similarity_indices]
            )

            if isinstance(most_common_label.mode, np.ndarray):
                most_similar_index = int(most_common_label.mode[0])

            else:
                most_similar_index = int(most_common_label.mode)

        else:
            most_similar_index = corresponding_labels[np.argmax(max_dot_products)]

        return dataset.images[most_similar_index]


# ==================================================================================================================================


if __name__ == "__main__":
    dataset1 = Dataset("datasets/cows")
    dataset2 = Dataset("datasets/cows_duplicates")

    fext = FeatureExtractor(dataset1, dataset2, "sift")

    fext.find_similars()
    print(fext.sift_similars)
