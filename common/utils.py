import os
import time
import sys
from colorama import Fore, Style, init

init()

colors = {
    "none": "",
    "status": Fore.LIGHTMAGENTA_EX,
    "done": Fore.GREEN,
    "fail": Fore.RED,
    "warning": Fore.YELLOW,
    "log": Fore.LIGHTBLACK_EX,
    "create": Fore.CYAN,
}  # Цвета текста

style = Style.BRIGHT

stamps = {
    "none": "",
    "status": "[%] ",
    "done": "[V] ",
    "fail": "[X] ",
    "warning": "[!] ",
    "log": "[$] ",
    "create": "[+] ",
}  # Штампы вывода

DATASET_PATH = "dataset/"

images = []


# ==================================================================================================================================
# |                                                            UTILS                                                               |
# ==================================================================================================================================


def get_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time() - start_time
        color_print("log", "log", f"Время выполнения: {end_time}", True)
        print("\n")
        return result

    return wrapper


# ==================================================================================================================================


def color_print(stamp: str, color: str, message: str, newline: bool) -> None:
    """## Функция для красивого вывода в консоль

    ## Input:

    ### stamp (string)
    - "none": ""
    - "status": [%]
    - "done": [V]
    - "fail": [X]
    - "warning": [!]
    - "log": [$]
    - "create": [+]

    ### color: (string)
    - "none": simple white
    - "status": lightmagenta
    - "done": green
    - "fail": red
    - "warning": yellow
    - "log": lightblack
    - "create": cyan

    ## using libraries: colorama, to install: "pip install colorama"
    """
    to_new_line = "\n" if newline else " "

    sys.stdout.write(f"{to_new_line}{style}{colors[color]}{stamps[stamp]}{message}")


# ==================================================================================================================================


def scan_directory(dataset_path: str) -> list[str]:
    """## Функция для сканирования директории и сохранения путей изображений.

    ### Input:
    - dataset_path (string): path to the dataset

    ### Output:
    - images path list"""
    try:
        for address, dirs, files in os.walk(dataset_path):
            for name in files:
                images.append(os.path.join(address, name))
    except Exception as e:
        color_print("fail", "fail", f"Error: {e}", "True")

    return images


# ==================================================================================================================================


# Проверка на скорость выполнения функции для сканирования директории
if __name__ == "__main__":
    get_time(scan_directory)()
