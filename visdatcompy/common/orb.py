import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from visdatcompy.common.utils import color_print
from visdatcompy.common.image_handler import Image, Dataset


__all__ = ["ORB"]


# ==================================================================================================================================
# |                                                              ORB TOOLS                                                         |
# ==================================================================================================================================


class ORB(object):
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        """
        Класс для поиска схожих изображений с помощью ORB.

        Attributes:
            - Dataset1 (Dataset): Объект класса Dataset.
            - Dataset2 (Dataset): Объект класса Dataset.

        Methods:
            - extract_features(dataset: Dataset): Извлекает дескрипторы из датасета
            и помещает их в объекты датасета.
            - find_similar_image(target_image: Image, dataset: Dataset): ищет схожие
            изображения в датасете.
            - visualize_similar_images(target_image: Image, dataset: Dataset): ищет
            схожие изображения и визуализирует найденную пару.
        """

        self.dataset1 = dataset1
        self.dataset2 = dataset2 if dataset2.path != dataset1.path else dataset1

    def extract_features(self, dataset: Dataset) -> tuple:
        """
        Извлекает дескрипторы и метки из изображений в датасете. Присваивает их
        полученному объекту класса Dataset.

        Parameters:
            - dataset (Dataset): Объект датасета класса Dataset.

        Returns:
            - tuple: Кортеж из двух элементов:
                - numpy.ndarray: Массив дескрипторов ORB.
                - numpy.ndarray: Массив меток изображений.
        """

        orb = cv2.ORB_create()

        descriptor_count = 0
        descriptors_array = np.zeros((10000000, 32))
        labels_array = np.zeros((10000000,))

        for i, image in enumerate(dataset.images):
            img = self._read_image_as_rgb(image)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            kp = orb.detect(img_gray, None)
            kp, des = orb.compute(img_gray, kp)

            current_descriptors_count = des.shape[0]

            for desc in range(current_descriptors_count):
                current_descriptor = des[desc, :]

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
            f"Количество дескрипторов ORB в {str(dataset.image_count)} изображениях: {str(descriptor_count)}",
        )

        dataset.orb_descriptors = extracted_descriptors
        dataset.orb_descriptors_labels = extracted_labels

        return extracted_descriptors, extracted_labels

    def find_similar_image(self, target_image: Image, dataset: Dataset) -> Image:
        """
        Находит наиболее похожее изображение для указанного с помощью дескрипторов ORB.

        Parameters:
            - target_image (Image): Объект целевого изображения (для поиска похожих на него).
            - dataset (Dataset): Объект датасета (для поиска схожих в нём).

        Returns:
            - Image: Объект наиболее похожего изображения.
        """

        # Получение индекса целевого изображения и индексов изображений из других классов
        target_image_index = dataset.filenames.index(target_image.filename)

        same_class_indices = np.where(
            dataset.orb_descriptors_labels == target_image_index
        )[0]
        different_class_indices = np.where(
            dataset.orb_descriptors_labels != target_image_index
        )[0]

        # Выделение дескрипторов целевого изображения и изображений других классов
        target_descriptors = dataset.orb_descriptors[same_class_indices, :]
        other_descriptors = dataset.orb_descriptors[different_class_indices, :]
        other_labels = dataset.orb_descriptors_labels[different_class_indices]

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
        color_print("create", "create", f"Чтение изображения: {image.filename}")

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

    # Создаём объект класса ORB для работы с двумя датасетами:
    orb = ORB(dataset1, dataset2)

    # Получаем дескрипторы для каждого датасета (они записываются в объекты Dataset)
    orb.extract_features(dataset1)
    orb.extract_features(dataset2)

    # Получаем объект изображения по названию
    my_image = dataset1.get_image("2_2.jpg")
    # Получаем изображение схожее с нашим из указанного датасета
    new_image: Image = orb.find_similar_image(my_image, dataset2)

    # Визуализируем оба изображения для визуальной оценки
    my_image.visualize()
    new_image.visualize()

    # Либо выполняем более простую функцию для поиска и отображения
    orb.visualize_similar_images(my_image, dataset2)
