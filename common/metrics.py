import os
import numpy as np
from PIL import Image
from utils import get_time


@get_time
def pix2pix(first_image_path: str, second_image_path: str) -> bool:
    """
    ## Функция для сравнения двух изображений методом Pixel to Pixel.

    ### Input:
    first_image_path (string): path to the first image
    second_image_path: (string): path to the second image

    ### Output:
    - image equal (boolean)
    """

    resized_image1 = Image.open(first_image_path).resize((512, 512))
    resized_image2 = Image.open(second_image_path).resize((512, 512))

    are_equal = np.array_equal(np.array(resized_image1), np.array(resized_image2))

    return are_equal


if __name__ == "__main__":
    print(pix2pix("dataset/000.jpg", "dataset/000.jpg"))
    print(pix2pix("dataset/000.jpg", "dataset/001.jpg"))
