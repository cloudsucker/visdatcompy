import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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
            - Dataset1 (Dataset): Объект класса Dataset.
            - Dataset2 (Dataset): Объект класса Dataset.
            - extractor (string): метод сравнения (sift, orb или fast).

        Methods:
            - extract_features(dataset: Dataset): Извлекает дескрипторы из датасета
            и помещает их в атрибуты датасета.
            - find_similar_image(target_image: Image, dataset: Dataset): ищет схожие
            изображения в датасете.
            - visualize_similar_images(target_image: Image, dataset: Dataset): ищет
            схожие изображения и визуализирует найденную пару.
        """

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

        self.dataset1 = dataset1
        self.dataset2 = dataset2 if dataset2.path != dataset1.path else dataset1

    def extract_features(self, dataset: Dataset, echo: bool = True) -> tuple:
        """
        Извлекает дескрипторы или метки из изображений в датасете. Присваивает их
        объекту класса Dataset.

        Parameters:
            - dataset (Dataset): Объект датасета класса Dataset.

        Returns:
            - tuple: Кортеж из двух элементов:
                - numpy.ndarray: Массив дескрипторов.
                - numpy.ndarray: Массив меток изображений.
        """

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

        return extracted_descriptors, extracted_labels

    def find_similar_image(self, target_image: Image, dataset: Dataset) -> Image:
        """
        Находит наиболее похожее изображение из датасета среди других.

        Parameters:
            - target_image (Image): Объект целевого изображения (для поиска похожих на него).
            - dataset (Dataset): Объект датасета (для поиска схожих в нём).

        Returns:
            - Image: Объект наиболее похожего изображения.
        """

        descriptors = getattr(dataset, self.extractor_name + "_descriptors")
        descriptors_labels = getattr(
            dataset, self.extractor_name + "_descriptors_labels"
        )

        try:
            # Получение индекса целевого изображения и индексов изображений из других классов
            target_image_index = dataset.filenames.index(target_image.filename)
        except ValueError:
            color_print("fail", "fail", "Датасет должен содержать целевое изображение.")
            return False

        same_class_indices = np.where(descriptors_labels == target_image_index)[0]
        different_class_indices = np.where(descriptors_labels != target_image_index)[0]

        # Выделение дескрипторов целевого изображения и изображений других классов
        target_descriptors = descriptors[same_class_indices, :]
        other_descriptors = descriptors[different_class_indices, :]
        other_labels = descriptors_labels[different_class_indices]

        # Вычисление попарного скалярного произведения между дескрипторами тестового и других изображений
        dot_products = np.dot(other_descriptors, target_descriptors.T)

        # Получение количества дескрипторов тестового изображения
        num_test_descriptors = target_descriptors.shape[0]

        # Создание массивов для хранения максимальных значений скалярного произведения и меток изображений
        max_dot_products = np.zeros((num_test_descriptors,))
        corresponding_labels = []

        # Выбор наиболее похожего изображения на основе скалярного произведения
        for k in range(num_test_descriptors):
            dot_product_values = dot_products[:, k]
            max_value = dot_product_values.max()
            max_dot_products[k] = max_value
            max_index = np.where(dot_product_values == max_value)
            corresponding_labels.append(other_labels[max_index[0]])

        # Определение наиболее часто встречающейся метки среди изображений с высокой похожестью
        high_similarity_indices = np.where(max_dot_products > 0.9)
        most_common_label = stats.mode(
            np.array(corresponding_labels)[high_similarity_indices]
        )

        most_similar_index = int(most_common_label.mode[0])

        return dataset.images[most_similar_index]

    def visualize_similar_images(self, target_image: Image, dataset: Dataset):
        """
        Находит и визуализирует целевое и наиболее похожее изображения.

        Parameters:
            - target_image (Image): Объект изображения класса Image.
            - dataset (Dataset): Объект датасета класса Dataset.
        """

        img = target_image._read_image_as_rgb()

        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.show()
        print(" ")

        ifound = self.find_similar_image(target_image, dataset)
        similar_image = ifound._read_image_as_rgb()

        color_print(
            "done",
            "done",
            f"Для изображения {target_image.filename}, наиболее похожее изображение: {ifound.filename}.",
        )

        plt.imshow(similar_image, cmap="gray")
        plt.axis("off")
        plt.show()


# ==================================================================================================================================


if __name__ == "__main__":
    dataset1 = Dataset("datasets/drone")
    dataset2 = Dataset("datasets/drone")

    fext = FeatureExtractor(dataset1, dataset2, "fast")

    fext.extract_features(dataset2)

    image = dataset2.get_image("0_1.jpg")

    fext.visualize_similar_images(image, dataset2)
