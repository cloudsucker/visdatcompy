import os
import cv2

from visdatcompy.common.utils import scan_directory, color_print


class Images(object):
    def __init__(self, dataset_path: str):
        """
        Класс изображений, объект класса представляет собой датасет, состоящий из изображений
        внутри указанного пути.

        Parameters:
            - dataset_path: путь к датасету.

        Attributes:
            - dataset_path: путь к датасету.
            - dataset_paths: пути к найденным изображениям в датасете.
            - dataset_name: название датасета.
            - dataset_image_count: кол-во найденных изображений в датасете.
        """
        self.dataset_path = dataset_path
        self.dataset_paths = self._scan_dataset_paths(self.dataset_path)
        self.dataset_name = self._dataset_small_paths[0][0]

        self.dataset_image_count = len(self.dataset_paths)

    def status(self):
        """
        Выводит основную информацию об объекте класса и его атрибутах:
            - dataset_path: Путь к датасету.
            - dataset_image_count: Количество полученых изображений.
        """
        color_print("done", "done", "Dataset & Images Status:")
        color_print("status", "log", "Путь к датасету:")
        color_print("none", "status", self.dataset_path, False)
        color_print("log", "log", "Количество изображений:")
        color_print("none", "create", self.dataset_image_count, False)
        print("\n")

    def visualize(self, image_name: str):
        """
        Открывает любую фотографию (фотографии) из датасета по её названию.
        Принимает неполные названия, ищет по вхождению в строках путей.
        При нахождении нескольких изображений открывает их поочереди.

        Parameters:
            - image_name (str): полное/неполное имя файла.
        """
        new_width = 800
        new_height = 600
        for image_path in self.dataset_paths:
            if image_name in image_path:
                image = cv2.imread(image_path)
                resized_image = cv2.resize(image, (new_width, new_height))
                cv2.imshow(image_name, resized_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def _scan_dataset_paths(self, dataset_path):
        dataset_small_paths = scan_directory(dataset_path)
        self._dataset_small_paths = dataset_small_paths

        dataset_paths = list(
            map(lambda x: os.path.join(x[0], x[1]), dataset_small_paths)
        )

        return dataset_paths


if __name__ == "__main__":
    ds1 = Images("dataset_small")
    ds1.status()
    ds1.visualize("5_3")
