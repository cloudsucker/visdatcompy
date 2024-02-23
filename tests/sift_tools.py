import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from common.utils import scan_directory


def num2fixstr(x, d):
    """
    Преобразует число в строку заданной длины, добавляя нули слева.

    Parameters:
        - x (int): число, которое нужно преобразовать в строку.
        - d (int): длина строки, которую нужно вернуть.

    Returns:
        - str: строка заданной длины.
    """

    st = "%0*d" % (d, x)

    return st


def ImageLoad(prefix, num_img, echo="off"):
    """
    Загружает изображение из файла.

    Parameters:
        - prefix (str): префикс имени файла изображения.
        - num_img (int): номер изображения.
        - echo (str, опционально): флаг, указывающий, нужно ли выводить информацию о загрузке изображения. По умолчанию выключен.

    Returns:
        - numpy.ndarray: загруженное изображение в формате RGB.
    """

    st = prefix + num2fixstr(num_img, 5) + ".png"

    if echo == "on":
        print("loading image " + st + "...")

    img = cv2.cvtColor(cv2.imread(st), cv2.COLOR_BGR2RGB)

    return img


def find_similar_image(itest, X, y):
    """
    Определяет наиболее похожее изображение на заданное изображение.

    Parameters:
        - itest (int): индекс тестируемого изображения.
        - X (numpy.ndarray): матрица признаков изображений.
        - y (numpy.ndarray): метки классов изображений.

    Returns:
        - int: индекс наиболее похожего изображения.
    """
    ik = itest - 1
    ii = np.where(y == ik)[0]
    jj = np.where(y != ik)[0]

    Xi = X[ii, :]
    Xj = X[jj, :]
    yj = y[jj]

    Dt = np.dot(Xj, Xi.T)

    n = Xi.shape[0]

    z = np.zeros((n,))
    d = np.zeros((n,))

    for k in range(n):
        h = Dt[:, k]
        i = h.max()
        z[k] = i
        j = np.where(h == i)
        d[k] = yj[j]

    kk = np.where(z > 0.9)

    m = stats.mode(d[kk])
    ifound = int(m[0]) + 1

    return ifound


def visualize_similar_images(itest):
    """
    Визуализирует тестовое изображение и его наиболее похожее изображение.

    Parameters:
        - itest (int): индекс тестируемого изображения.
    """

    ifound = find_similar_image(itest, X, y)
    I = ImageLoad("cows_dataset/V", itest, echo="on")

    print("Тестовое изображение: " + str(itest))
    plt.imshow(I, cmap="gray")
    plt.axis("off")
    plt.show()
    print(" ")
    J = ImageLoad("cows_dataset/V", ifound, echo="on")
    print(
        "Для изображения "
        + str(itest)
        + ", наиболее похожее изображение: "
        + str(ifound)
        + "."
    )
    plt.imshow(J, cmap="gray")
    plt.axis("off")
    plt.show()
    print("--------------------------------------------------")


if __name__ == "__main__":
    print(num2fixstr(3, 5))
    print(scan_directory("tests/cows_dataset/"))
