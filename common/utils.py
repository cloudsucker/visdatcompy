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


def get_time(func) -> None:
    """
    Декоратор для замера времени выполнения функции.

    Вход:
    - func (function): функция, время выполнения которой требуется замерить

    Вывод:
    - None
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time() - start_time
        color_print("log", "log", f"Время выполнения: {end_time}", True)
        return result

    return wrapper


# ==================================================================================================================================


def color_print(stamp: str, color: str, message: str, newline: bool) -> None:
    """
    Функция для красивого вывода в консоль.

    Вход:
    - stamp (str): штамп для определения вида сообщения
    - color (str): цвет текста сообщения
    - message (str): текст сообщения
    - newline (bool): флаг, определяющий, нужно ли добавлять переход на новую строку

    Вывод:
    - None

    Используемые штампы:
        - "none": пустой штамп
        - "status": [%]
        - "done": [V]
        - "fail": [X]
        - "warning": [!]
        - "log": [$]
        - "create": [+]

    Используемые цвета:
        - "none": обычный белый цвет
        - "status": светло-пурпурный цвет
        - "done": зеленый цвет
        - "fail": красный цвет
        - "warning": желтый цвет
        - "log": светло-черный цвет
        - "create": голубой цвет

    Используемые библиотеки:
        - colorama, установка: "pip install colorama"
    """

    to_new_line = "\n" if newline else " "

    sys.stdout.write(f"{to_new_line}{style}{colors[color]}{stamps[stamp]}{message}")


# ==================================================================================================================================


def scan_directory(dataset_path: str) -> list[str]:
    """
    Функция для сканирования директории и сохранения путей изображений.

    Вход:
    - dataset_path (str): путь к директории с изображениями

    Вывод:
    - images (list[str]): список путей к изображениям
    """

    try:
        for address, dirs, files in os.walk(dataset_path):
            for name in files:
                images.append((address, name))
    except Exception as e:
        color_print("fail", "fail", f"Error: {e}", "True")

    return images


# ==================================================================================================================================


# Проверка на скорость выполнения функции для сканирования директории
if __name__ == "__main__":
    get_time(scan_directory)("dataset")
