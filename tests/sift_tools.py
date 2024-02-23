import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from scipy import stats
from common.utils import color_print, scan_directory


# ==================================================================================================================================
# |                                                             SIFT TOOLS                                                         |
# ==================================================================================================================================


def load_image(image_path: str, echo: bool = False) -> np.ndarray:
    """
    Загружает изображение из файла и масштабирует его до ширины 512 пикселей, сохраняя пропорции.

    Parameters:
        - image_path (str): Путь к файлу изображения.
        - echo (str, optional): Управляет выводом сообщения о загрузке изображения. Если "on", то выводится сообщение.

    Returns:
        - numpy.ndarray: Загруженное и масштабированное изображение в формате RGB.
    """

    if echo == True:
        color_print("create", "create", f"Loading image {image_path}", True)

    # Загрузка изображения и преобразование его в формат RGB
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Получение текущих размеров изображения
    height, width = image.shape[:2]

    # Вычисление новых размеров, сохраняя пропорции
    new_width = 512
    new_height = int(height * (new_width / width))

    # Масштабирование изображения до новых размеров
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


# ==================================================================================================================================


def get_descriptors(dataset_path: str, echo: bool = False):
    """
    Извлекает дескрипторы SIFT из изображений в указанной директории.

    Parameters:
        - dataset_path (str): Путь к директории с изображениями.
        - echo (bool, optional): Управляет выводом информационных сообщений. Если установлено значение True, сообщения будут выводиться.

    Returns:
        - numpy.ndarray: Массив дескрипторов SIFT.
        - numpy.ndarray: Массив меток изображений.
    """

    # Создание объекта для извлечения дескрипторов SIFT
    feature_extractor = cv2.SIFT_create()

    # Инициализация переменных для хранения дескрипторов и меток
    t = 0
    Xt = np.zeros((10000000, 128))
    yt = np.zeros((10000000,))

    # Получение списка путей к изображениям в указанной директории
    image_paths = scan_directory(dataset_path)
    image_full_paths = list(map(lambda x: os.path.join(x[0], x[1]), image_paths))

    # Получение количества изображений в директории
    images_count = len(image_full_paths)

    # Вывод списка путей к изображениям, если установлен флаг echo
    print(image_full_paths) if echo else None

    # Извлечение дескрипторов SIFT из каждого изображения
    for i in range(images_count):
        image = load_image(image_full_paths[i], echo)
        J = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Извлечение дескрипторов SIFT
        kp, desc = feature_extractor.detectAndCompute(J, None)
        ni = desc.shape[0]
        for j in range(ni):
            f = desc[j, :]
            Xt[t, :] = f / np.linalg.norm(f)
            yt[t] = i
            t = t + 1

        # Вывод сообщения о количестве извлеченных дескрипторов, если установлен флаг echo
        if echo:
            color_print("done", "done", f"{str(ni)} Дескрипторы были извлечены.")

    # Обрезка массивов дескрипторов и меток до фактического размера
    X = Xt[0:t, :]
    y = yt[0:t]

    # Вывод сообщения о количестве извлеченных дескрипторов, если установлен флаг echo
    if echo:
        color_print(
            "done",
            "done",
            f"Количество дескрипторов SIFT в {str(images_count)} images: {str(t)}",
            True,
        )

    return X, y


# ==================================================================================================================================


def find_similar_image(itest, X, y):
    """
    Находит наиболее похожее изображение на тестовом изображении с помощью дескрипторов SIFT.

    Parameters:
        itest (int): Индекс тестового изображения.
        X (numpy.ndarray): Массив дескрипторов SIFT.
        y (numpy.ndarray): Массив меток изображений.

    Returns:
        int: Индекс наиболее похожего изображения.
    """

    # Определение индекса тестового изображения и индексов изображений из других классов
    ik = itest
    ii = np.where(y == ik)[0]
    jj = np.where(y != ik)[0]

    # Выделение дескрипторов тестового изображения и изображений других классов
    Xi = X[ii, :]
    Xj = X[jj, :]
    yj = y[jj]

    # Вычисление попарного скалярного произведения между дескрипторами тестового и других изображений
    Dt = np.dot(Xj, Xi.T)

    # Получение количества дескрипторов тестового изображения
    n = Xi.shape[0]

    # Создание массивов для хранения максимальных значений скалярного произведения и меток изображений
    z = np.zeros((n,))
    d = np.zeros((n,))

    # Выбор наиболее похожего изображения на основе скалярного произведения
    for k in range(n):
        h = Dt[:, k]
        i = h.max()
        z[k] = i
        j = np.where(h == i)
        d[k] = yj[j]

    # Определение наиболее часто встречающейся метки среди изображений с высокой похожестью
    kk = np.where(z > 0.9)
    m = stats.mode(d[kk])
    ifound = int(m[0])

    return ifound


# ==================================================================================================================================


def visualize_similar_images(itest):
    """
    Визуализирует тестовое и наиболее похожее изображения.

    Parameters:
        itest (int): Индекс тестового изображения.
    """

    # Нахождение наиболее похожего изображения на тестовом изображении
    ifound = find_similar_image(itest, X, y)

    # Загрузка и визуализация тестового изображения
    test_image = load_image(image_full_paths[itest], True)

    print("Тестовое изображение: " + str(itest))

    plt.imshow(test_image, cmap="gray")
    plt.axis("off")
    plt.show()
    print(" ")

    # Загрузка и визуализация наиболее похожего изображения
    similar_image = load_image(image_full_paths[ifound], True)

    print(
        "Для изображения "
        + str(itest)
        + ", наиболее похожее изображение: "
        + str(ifound)
        + "."
    )

    plt.imshow(similar_image, cmap="gray")
    plt.axis("off")
    plt.show()

    print("--------------------------------------------------")


# ==================================================================================================================================

if __name__ == "__main__":
    print("gdgd")
