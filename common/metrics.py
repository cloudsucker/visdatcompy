import os
import numpy as np
from PIL import Image
from utils import scan_directory, get_time


@get_time
def pix2pix(current_image_path: str, second_image_path: str) -> bool:
    """## Функция для сравнения двух изображений методом Pixel to Pixel.

    ### Input:
    current_image_path (string): path to the first image
    second_image_path: str

    ### Output:
    - image equal dict to current_image_path (boolean)
    example of output: {image_name, True/False}
    """
    image_pathlist = scan_directory(dataset_path)

    duplicates = {}

    resized_image1 = Image.open(current_image_path).resize((512, 512))

    for image in image_pathlist:
        image_name = os.path.basename(image)
        second_image = Image.open(image)
        resized_image2 = np.array(second_image.resize((512, 512)))

        are_equal = np.array_equal(np.array(resized_image1), np.array(resized_image2))

    return duplicates
