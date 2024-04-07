import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from visdatcompy.common.utils import color_print, scan_directory

# ==================================================================================================================================
# |                                                             SIFT TOOLS                                                         |
# ==================================================================================================================================


def load_image(image_path: str, echo: bool = False) -> np.ndarray:
    """
    Загружает изображение из файла и масштабирует его до ширины 512 пикселей, сохраняя пропорции.

    Parameters:
        - image_path (str): Путь к файлу изображения.
        - echo (bool, optional): Управляет выводом сообщения о загрузке изображения. Если True, то выводится сообщение.

    Returns:
        - numpy.ndarray: Загруженное и масштабированное изображение в формате RGB.
    """

    if echo:
        color_print("create", "create", f"Загружаем изображение {image_path}", True)

    # Загрузка изображения и преобразование его в формат RGB
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Получение текущих размеров изображения
    height, width = image.shape[:2]

    # Вычисление новых размеров, сохраняя пропорции
    new_width = 256  # изменил с 512 до 256
    new_height = int(height * (new_width / width))

    # Масштабирование изображения до новых размеров
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


# ==================================================================================================================================


def get_descriptors(dataset_path: str, echo: bool = False) -> tuple:
    """
    Извлекает дескрипторы SIFT из изображений в указанной директории.

    Parameters:
        - dataset_path (str): Путь к директории с изображениями.
        - echo (bool, optional): Управляет выводом информационных сообщений. Если установлено значение True, сообщения будут выводиться.

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

    # Получение списка путей к изображениям в указанной директории
    image_paths = scan_directory(dataset_path)
    image_full_paths = list(map(lambda x: os.path.join(x[0], x[1]), image_paths))

    # Получение количества изображений в директории
    images_count = len(image_full_paths)

    # Вывод списка путей к изображениям, если установлен флаг echo
    color_print("log", "log", image_full_paths, True) if echo else None

    # Извлечение дескрипторов SIFT из каждого изображения
    for i in range(images_count):
        image = load_image(image_full_paths[i], echo)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = feature_extractor.detectAndCompute(image_gray, None)
        current_descriptors_count = descriptors.shape[0]

        # Цикл для обработки каждого дескриптора
        for j in range(current_descriptors_count):
            # Получение j-го дескриптора из общего массива дескрипторов
            current_descriptor = descriptors[j, :]
            # Нормализация дескриптора и запись его в массив descriptors_array
            descriptors_array[descriptor_count, :] = (
                current_descriptor / np.linalg.norm(current_descriptor)
            )
            # Присвоение метки i для текущего дескриптора в массиве labels_array
            labels_array[descriptor_count] = i
            # Увеличение счетчика дескрипторов
            descriptor_count += 1

        # Вывод сообщения о количестве извлеченных дескрипторов, если установлен флаг echo
        if echo:
            color_print(
                "done",
                "done",
                f"{str(current_descriptors_count)} дескрипторов было извлечено.",
                True,
            )

    # Обрезка массивов дескрипторов и меток до фактического размера
    extracted_descriptors = descriptors_array[0:descriptor_count, :]
    extracted_labels = labels_array[0:descriptor_count]

    # Вывод сообщения о количестве извлеченных дескрипторов, если установлен флаг echo
    if echo:
        color_print(
            "done",
            "done",
            f"Количество дескрипторов SIFT в {str(images_count)} изображениях: {str(descriptor_count)}",
            True,
        )

    return extracted_descriptors, extracted_labels


# ==================================================================================================================================


def find_similar_image(
    test_image_index: int, descriptors_array: np.ndarray, labels_array: np.ndarray
) -> int:
    """
    Находит наиболее похожее изображение на тестовом изображении с помощью дескрипторов SIFT.

    Parameters:
        - test_image_index (int): Индекс тестового изображения.
        - descriptors_array (numpy.ndarray): Массив дескрипторов SIFT.
        - labels_array (numpy.ndarray): Массив меток изображений.

    Returns:
        - int: Индекс наиболее похожего изображения.
    """

    # Определение индекса тестового изображения и индексов изображений из других классов
    test_index = test_image_index
    same_class_indices = np.where(labels_array == test_index)[0]
    different_class_indices = np.where(labels_array != test_index)[0]

    # Выделение дескрипторов тестового изображения и изображений других классов
    test_descriptors = descriptors_array[same_class_indices, :]
    other_descriptors = descriptors_array[different_class_indices, :]
    other_labels = labels_array[different_class_indices]

    # Вычисление попарного скалярного произведения между дескрипторами тестового и других изображений
    dot_products = np.dot(other_descriptors, test_descriptors.T)

    # Получение количества дескрипторов тестового изображения
    num_test_descriptors = test_descriptors.shape[0]

    # Создание массивов для хранения максимальных значений скалярного произведения и меток изображений
    max_dot_products = np.zeros((num_test_descriptors,))
    corresponding_labels = np.zeros((num_test_descriptors,))

    # Выбор наиболее похожего изображения на основе скалярного произведения
    for k in range(num_test_descriptors):
        dot_product_values = dot_products[:, k]
        max_value = dot_product_values.max()
        max_dot_products[k] = max_value
        max_index = np.where(dot_product_values == max_value)
        corresponding_labels[k] = other_labels[max_index]

    # Определение наиболее часто встречающейся метки среди изображений с высокой похожестью
    high_similarity_indices = np.where(max_dot_products > 0.9)
    most_common_label = stats.mode(corresponding_labels[high_similarity_indices])
    most_similar_index = int(most_common_label[0])

    return most_similar_index


# ==================================================================================================================================


def visualize_similar_images(
    itest: int, X: np.ndarray, y: np.ndarray, image_full_paths: list
):
    """
    Визуализирует тестовое и наиболее похожее изображения.

    Parameters:
        - itest (int): Индекс тестового изображения.
        - X (numpy.ndarray): Массив дескрипторов SIFT.
        - y (numpy.ndarray): Массив меток изображений.
        - image_full_paths (list): Список полных путей к изображениям.
    """

    # Нахождение наиболее похожего изображения на тестовом изображении
    ifound = find_similar_image(itest, X, y)

    # Загрузка и визуализация тестового изображения
    test_image = load_image(image_full_paths[itest], True)

    color_print("status", "status", f"Тестовое изображение: {str(itest)}", True)

    plt.imshow(test_image, cmap="gray")
    plt.axis("off")
    plt.show()
    print(" ")

    # Загрузка и визуализация наиболее похожего изображения
    similar_image = load_image(image_full_paths[ifound], True)

    color_print(
        "done",
        "done",
        f"Для изображения {str(itest)}, наиболее похожее изображение: {str(ifound)}.",
        True,
    )

    plt.imshow(similar_image, cmap="gray")
    plt.axis("off")
    plt.show()

    print("--------------------------------------------------")


# ==================================================================================================================================

r"""
WARNING: На данный момент визуализируется только одно изображение, после чего надо перезапускать весь код и заново собирать все дескрипторы.

TODO: Сделать для функции scan_directory флаг для возвращения полных или неполных путей в зависимости от выбора.

TODO: Оформить всё это в виде класса с сохранением всех дескрипторов и функциями для отображения результатов.

FIXME: Убрать "RuntimeWarning" в common.metrics для метода PSNR.

FIXME: После вывода общего кол-ва дескрипторов выводится предупреждение: 
[V] Количество дескрипторов SIFT в 322 изображениях: 318017c:\Users\sharj\Desktop\Учёба\visdatcompy\tests\sift_tools.py:167: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
corresponding_labels[k] = other_labels[max_index]
"""

if __name__ == "__main__":
    # Определение пути к директории с изображениями
    dataset_path = "datasets/drones"

    time_start = time.time()

    # Извлечение дескрипторов SIFT из изображений
    X, y = get_descriptors(dataset_path, echo=True)

    time_end = time.time()

    color_print(
        "log",
        "log",
        f"Время извлечения дескриптеров SIFT из 500 изображений: {time_end - time_start} секунд.",
    )

    # Получение списка путей к изображениям в директории
    image_paths = scan_directory(dataset_path)

    # Формирование полных путей к изображениям
    image_full_paths = list(map(lambda x: os.path.join(x[0], x[1]), image_paths))

    time_start = time.time()

    # Визуализация тестового изображения и его наиболее похожего изображения
    visualize_similar_images(0, X, y, image_full_paths)

    time_end = time.time()

    color_print(
        "log",
        "log",
        f"Время поиска совпадения для 1 изображения из 500 изображений: {time_end - time_start} секунд.",
    )
