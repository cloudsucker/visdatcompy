from matplotlib import pyplot as plt
import numpy as np
from ipywidgets import interact
from visdatcompy.common.SIFT.sift_tools import find_similar_image, load_image
class SIFT_GUI:
    def __init__(self, X = None, y = None, image_full_paths = None):
        self.X = X
        self.y = y
        self.image_full_paths = image_full_paths
    def interactive_view(self):
        interact(self.visualize_similar_images, itest=(0, len(self.image_full_paths)))
    def visualize_similar_images(
        #self, itest: int, X: np.ndarray, y: np.ndarray, image_full_paths: list
        self, itest: int
    ):
        """
        Визуализирует тестовое и наиболее похожее изображения.

        Parameters:
            - itest (int): Индекс тестового изображения.
            - X (numpy.ndarray): Массив дескрипторов SIFT.
            - y (numpy.ndarray): Массив меток изображений.
            - image_full_paths (list): Список полных путей к изображениям.
        """
        ifound = find_similar_image(itest, self.X, self.y)
        I = load_image(self.image_full_paths[itest],echo='on')
        print('Тестовое изображение: '+str(itest))

        plt.imshow(I,cmap='gray')
        plt.axis('off')
        plt.show()
        print(' ')

        J = load_image(self.image_full_paths[ifound],echo='on')
        print('Для изображения '+str(itest)+', наиболее похожее изображение: '+str(ifound)+'.')
        plt.imshow(J,cmap='gray')
        plt.axis('off')
        plt.show()
        print('--------------------------------------------------')

