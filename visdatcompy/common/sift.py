import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from visdatcompy.common.utils import color_print
from visdatcompy.common.image_handler import Image, Dataset


__all__ = ["SIFT"]


# ==================================================================================================================================
# |                                                             SIFT TOOLS                                                         |
# ==================================================================================================================================


class SIFT:
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        """
        Класс для поиска схожих изображений с помощью SIFT.

        Attributes:
            - Dataset1 (Dataset): Объект класса Dataset.
            - Dataset2 (Dataset): Объект класса Dataset.

        Methods:
            - get_descriptors(dataset: Dataset): Извлекает дескрипторы из датасета
            и помещает их в объекты датасета.
            - find_similar_image(target_image: Image, dataset: Dataset): ищет схожие
            изображения в датасете.
            - visualize_similar_images(target_image: Image, dataset: Dataset): ищет
            схожие изображения и визуализирует найденную пару.
        """

        self.Dataset1 = dataset1
        self.Dataset2 = dataset2 if dataset1.path != dataset2.path else dataset1

    def get_descriptors(self, dataset: Dataset) -> tuple:
        """
        Извлекает дескрипторы SIFT из изображений в датасете. Присваивает их
        объекту класса Dataset.

        Parameters:
            - dataset (Dataset): Объект датасета класса Dataset.

        Returns:
            - tuple: Кортеж из двух элементов:
                - numpy.ndarray: Массив дескрипторов SIFT.
                - numpy.ndarray: Массив меток изображений.
        """

        # Создание объекта для извлечения дескрипторов SIFT
        feature_extractor = cv2.SIFT_create()

        # Инициализация переменных для хранения дескрипторов и меток
        descriptor_count = 0
        descriptors_array = np.zeros((10000000, 128))
        labels_array = np.zeros((10000000,))

        # Извлечение дескрипторов SIFT из каждого изображения
        for i, image in enumerate(dataset.images):

            img = self._read_image_as_rgb(image)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            _, descriptors = feature_extractor.detectAndCompute(img_gray, None)
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

        color_print(
            "done",
            "done",
            f"Количество дескрипторов SIFT в {str(dataset.image_count)} изображениях: {str(descriptor_count)}",
        )

        dataset.descriptors = extracted_descriptors
        dataset.descriptors_labels = extracted_labels

        return extracted_descriptors, extracted_labels

    def find_similar_image(self, target_image: Image, dataset: Dataset) -> Image:
        """
        Находит наиболее похожее изображение для указанного с помощью дескрипторов SIFT.

        Parameters:
            - target_image (Image): Объект целевого изображения (для поиска похожих на него).
            - dataset (Dataset): Объект датасета (для поиска схожих в нём).

        Returns:
            - Image: Метка наиболее похожего изображения.
        """

        # Получение индекса целевого изображения и индексов изображений из других классов
        target_image_index = dataset.filenames.index(target_image.filename)

        same_class_indices = np.where(dataset.descriptors_labels == target_image_index)[
            0
        ]
        different_class_indices = np.where(
            dataset.descriptors_labels != target_image_index
        )[0]

        # Выделение дескрипторов целевого изображения и изображений других классов
        target_descriptors = dataset.descriptors[same_class_indices, :]
        other_descriptors = dataset.descriptors[different_class_indices, :]
        other_labels = dataset.descriptors_labels[different_class_indices]

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

        img = self._read_image_as_rgb(target_image)

        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.show()
        print(" ")

        ifound = self.find_similar_image(target_image, dataset)
        similar_image = self._read_image_as_rgb(ifound)

        color_print(
            "done",
            "done",
            f"Для изображения {target_image.filename}, наиболее похожее изображение: {ifound.filename}.",
        )

        plt.imshow(similar_image, cmap="gray")
        plt.axis("off")
        plt.show()

    def _read_image_as_rgb(self, image: Image) -> np.ndarray:
        color_print("create", "create", f"Загружаем изображение {image.filename}")

        rgb_image = cv2.cvtColor(image.read(), cv2.COLOR_BGR2RGB)

        new_width = 512
        new_height = int(image.height * (new_width / image.width))

        resized_image = cv2.resize(rgb_image, (new_width, new_height))

        return resized_image


# ==================================================================================================================================


if __name__ == "__main__":
    # Создаём два объекта с датасетами
    dataset1 = Dataset(r"datasets\small_drone_test_compressed")
    dataset2 = Dataset(r"datasets\small_drone_test_compressed")

    # Создаём объект класса SIFT для работы с двумя датасетами:
    sift = SIFT(dataset1, dataset2)

    # Получаем дескрипторы для каждого датасета (они записываются в объекты Dataset)
    sift.get_descriptors(dataset1)
    sift.get_descriptors(dataset2)

    # Получаем объект изображения по названию
    my_image = dataset1.get_image("2_2.jpg")
    # Получаем изображение схожее с нашим из указанного датасета
    new_image: Image = sift.find_similar_image(my_image, dataset2)

    # Визуализируем оба изображения для визуальной оценки
    my_image.visualize()
    new_image.visualize()

    # Либо выполняем более простую функцию для поиска и отображения
    sift.visualize_similar_images(my_image, dataset2)
