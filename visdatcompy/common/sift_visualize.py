from ipywidgets import interact
from matplotlib import pyplot as plt

from visdatcompy.common.utils import color_print
from visdatcompy.common.sift import find_similar_image, load_image


class SIFT_GUI:
    def __init__(self, descriptors=None, y=None, image_full_paths=None):
        self.descriptors = descriptors
        self.y = y
        self.image_full_paths = image_full_paths

    def interactive_view(self):
        interact(self.visualize_similar_images, itest=(0, len(self.image_full_paths)))

    def visualize_similar_images(self, itest: int):
        """
        Визуализирует тестовое и наиболее похожее изображения.

        Parameters:
            - itest (int): Индекс тестового изображения.
            - descriptors (numpy.ndarray): Массив дескрипторов SIFT.
            - y (numpy.ndarray): Массив меток изображений.
            - image_full_paths (list): Список полных путей к изображениям.
        """

        ifound = find_similar_image(itest, self.descriptors, self.y)
        image = load_image(self.image_full_paths[itest], echo="on")

        color_print("log", "log", f"Тестовое изображение: {str(itest)}")

        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()
        print(" ")

        J = load_image(self.image_full_paths[ifound], echo="on")

        color_print(
            "log",
            "log",
            f"Для изображения {str(itest)}, наиболее похожее изображение: {str(ifound)}.",
        )

        plt.imshow(J, cmap="gray")
        plt.axis("off")
        plt.show()
