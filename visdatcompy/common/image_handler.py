import os
import cv2
import numpy as np
import pandas as pd
from PIL.ExifTags import TAGS
from PIL import Image as PIL_Image

from visdatcompy.common.utils import color_print


__all__ = ["Image", "Dataset"]


# ==================================================================================================================================
# |                                                          IMAGE HANDLER                                                         |
# ==================================================================================================================================


class Image(object):
    def __init__(self, image_path: str):
        self.path = image_path

        self.filename = os.path.basename(self.path)

    def info(self):
        self.height, self.width, self.channel = self.read().shape

        color_print("done", "done", "Image Information:")

        color_print("log", "log", "Название изображения:")
        color_print("none", "status", self.filename, False)

        color_print(
            "log",
            "log",
            "Разрешение изображения:",
        )
        color_print(
            "none", "status", f"{self.width} x {self.height} x {self.channel}", False
        )

    def read(self):
        image = cv2.imread(self.path)
        self.height, self.width, self.channel = image.shape

        if image is None:
            color_print("fail", "fail", f"Ошибка чтения изображения: {self.image_name}")

        return image

    def read_and_resize(self):
        """
        Читает изображение, изменяет его размер на 512x512 и возвращает
        его в виде одномерного NumPy массива.

        Returns:
            - img_array: NumPy-массив открытого изображения.
        """

        with PIL_Image.open(self.path) as image:
            image_resized = image.resize((512, 512))
            image_array = np.array(image_resized)

        return image_array.flatten()

    def visualize(self):
        image = self.read()

        new_width = 800
        new_height = 600

        resized_image = cv2.resize(image, (new_width, new_height))

        cv2.imshow(self.filename, resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_exif_data(self) -> dict[str, str]:
        """
        Получение метаданных изображения.

        Creates:
            - exif_data (dict[str, str]): словарь с метаданными изображения.
        """

        exif_dict = {}

        try:
            with PIL_Image.open(self.path) as image:
                exif_data = image._getexif()

                if exif_data:
                    for tag, value in exif_data.items():
                        tag_name = TAGS.get(tag, tag)
                        exif_dict[tag_name] = value
                else:
                    color_print("fail", "fail", "exif-данные не найдены.")

                    return None

        except FileNotFoundError:
            color_print("fail", "fail", "Изображение не найдено.")
            return None

        self.exif_data = exif_dict

        return exif_dict


# ==================================================================================================================================


class Dataset(object):
    def __init__(self, dataset_path: str):
        """
        Класс датасета, объект класса представляет собой датасет, состоящий из изображений
        внутри указанного пути.

        Parameters:
            - dataset_path: путь к датасету (или изобрежению для создания датасета с одним изображением).

        Attributes:
            - path: путь к датасету.
            - name: название датасета (имя папки).
            - images (list[Image]): список с объектами изображений (объектов класса Image).
            - image_generator: генератор для получения объектов изображений по одному.
            - image_count: кол-во найденных изображений в датасете.
        """

        self.path = dataset_path
        self.name = os.path.basename(self.path)

        self.images: list[Image]
        self.filenames, self.images = self._get_images()
        self.image_generator = self._image_generator()

        self.image_count = len(self.images)

    def info(self):
        """
        Выводит основную информацию об объекте класса и его атрибутах:
            - path: Путь к датасету.
            - image_count: Количество полученых изображений.
        """

        color_print("done", "done", "Dataset Information:")

        color_print("status", "log", "Название датасета:")
        color_print("none", "status", self.name, False)

        color_print("status", "log", "Количество изображений:")
        color_print("none", "create", self.image_count, False)

        color_print("status", "log", "Путь к датасету:")
        color_print("none", "status", self.path, False)

        print("\n")

    def get_image(self, filename: str) -> Image:
        """
        Возвращает объект изображения по названию файла.

        Parameters:
            - filename (str): Имя изображения с его расширением.

        Returns:
            - Image: объект изображения с указаным именем файла.
        """

        image_index = self.filenames.index(filename)
        return self.images[image_index]

    def get_exif_data(self) -> list[str]:
        """
        Собирает метаданные со всех изображений в датасете и сохраняет их в атрибут exif_data.

        Creates:
            - exif_data (pd.DataFrame): датафрейм с метаданными датасета.
        """

        exif_data = []

        for image in self.images:
            image_exif_data = image.get_exif_data()

            if image_exif_data is not None:
                filename, file_extension = os.path.splitext(str(image.filename))
                image_exif_data.update(
                    {"Filename": filename, "FileExtension": file_extension}
                )

                exif_data.append(image_exif_data)

        exif_df = pd.DataFrame(data=exif_data)
        exif_df = exif_df.drop(columns="MakerNote")

        self.exif_data = exif_df

    def _get_images(self):
        filenames = []
        images = []

        if os.path.isdir(self.path):
            for image_name in os.listdir(self.path):
                image = Image(os.path.join(self.path, image_name))
                filenames.append(image.filename)
                images.append(image)

        elif os.path.isfile(self.path):
            image = Image(self.path)
            filenames.append(image.filename)
            images.append(image)

        return filenames, images

    def _image_generator(self):
        """
        Генератор для итерации по изображениям в датасете. Возвращает по одному объекту изображения за шаг.

        Returns:
            - Image (object) - объект изображения.
        """

        for image in self.images:
            yield image


# ==================================================================================================================================


if __name__ == "__main__":
    # Создаём объект класса Image для работы с одним изображением:
    img = Image(r"datasets\drone\000.jpg")

    # Выводим информацию об изображении и открываем его
    img.info()
    img.visualize()

    # Создаём объект класса Dataset для работы с датасетом.
    dataset = Dataset(r"datasets\drone")

    # Выводим информацию о датасете
    dataset.info()

    # Получаем объект изображения из датасета по имени файла
    # и считываем его сжимая до 512x512
    flatten_image = dataset.get_image("000.jpg").read_and_resize()
