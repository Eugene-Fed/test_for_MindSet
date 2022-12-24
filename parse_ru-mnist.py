# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import re
# import math                           # не используем, т.к. не режем широкие распознанные блоки
from PIL import Image

# Список всех настроечных параметров/констант
WORK_DIR = r"D:\work\test_comp_vision\datasets\!_lines_w25"
EXPORT_DIR = r"D:\work\test_comp_vision\datasets\!_lines_w25_parsed_fix"
OUT_SIZE = 28  # размер выходных изображений
LIMIT_SIZE = 8  # размер блоков на изображении, меньше которого текст не вырезается
SYMBOL_DIVIDE = 1.2  # если ширина блока больше высоты на этот коэффициент - то разделить его пополам
PATTERN = r'\w'  # шаблон по которому будем извлекать из текста символы (только буквы и цифры)
START_INDEX = 0  # индекс, с которого продолжаем обрабатывать файлы. Not used yet
END_INDEX = 10
LABELS = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'


# Функция для получения списка файлов из каталога с фотографиями (как в task_1 и task_2)
def get_files(directory: str) -> list:
    # TODO - переписать под использование Pathlib
    paths = []  # полный пути к файлам изображений
    names = []  # имя без расширения
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            paths.append(os.path.join(directory, filename))
            names.append(filename.split('.')[0])

    return paths, names


def resize_image(image, out_size: int):
    h, w = image.shape[0], image.shape[1]
    h_scale, w_scale = out_size / h, out_size / w
    new_h = int(h * h_scale)
    new_w = int(w * w_scale)
    return cv2.resize(image, (new_w, new_h))


def scale_image(image, scale):  # принимаем объект изображения OpenCV
    # получаем текущий размер, вычисляем искомый и создаем измененное изображение
    height, width = image.shape[0], image.shape[1]
    img_width = int(width * scale)
    img_height = int(height * scale)
    img = cv2.resize(image, (img_width, img_height))
    # img = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def crop_resize_letters(img, y, h, x, w):
    y -= 2          # чтобы не потерять точки и крышечку в Ё b Й
    h += 2
    w += 1          # чтобы не потерять палочку в Ы
    letter_crop = img[y:y + h, x:x + w]

    # Resize letter canvas to square
    size_max = max(w, h)
    # TODO - Изменить, чтобы вместо заполнения белым фоном, изображения расширялись до квадрата
    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)

    try:
        if w > h:
            y_pos = size_max // 2 - h // 2
            letter_square[y_pos:y_pos + h, 0:w] = letter_crop
        elif w < h:
            x_pos = size_max // 2 - w // 2
            letter_square[0:h, x_pos:x_pos + w] = letter_crop
        else:
            letter_square = letter_crop
    except ValueError as ve:
        print(ve)
        letter_square = scale_image(letter_crop, OUT_SIZE)

    return letter_square


def extract_letters(file_path: str, out_size=OUT_SIZE) -> list:
    # out_size - задает размер к которому будет приведено изображение
    img = cv2.imread(file_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # преобразуем в ЧБ
    else:
        gray = img

    kernel = np.ones((2, 2), 'uint8')
    # erosion = cv2.erode(gray, kernel, iterations=1)
    # thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = 255 - thresh

    result = invert

    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # output = img.copy()
    # cv2.waitKey(0)

    letters = []
    # blocks = []         # чисто для тестового вывода изображений
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
        limit = int(LIMIT_SIZE)  # ограничение на мелкие символы, чтобы их не вырезать
        if limit < h < img.shape[0] and limit < w < img.shape[1]:  # игнорируем маленькие блоки размером с изображение
            if w < h * SYMBOL_DIVIDE:  # если ширина найденного блока не слишком велика, то вырезаем символ
                letters.append((x, w, cv2.resize(crop_resize_letters(img=gray, y=y, h=h, x=x, w=w),
                                                 (out_size, out_size), interpolation=cv2.INTER_AREA)))
                # blocks.append((x, w, result[y:y + h, x:x + w]))
            else:
                return []           # если есть потенциальные ошибки распознавания - возвращаем пустой список
                break  # Убрал обработку широких символов, чтобы минимизировать ошибки в сопоставлении данных

                """
                symbol_count = math.ceil(w / h)  # округляем символы до большего целого
                w = math.floor(w / symbol_count)  # округляем ширину в пикселях до меньшего целого
                for i in range(symbol_count):
                    x += i * w
                    # Resize letter to 28x28 and add letter and its X-coordinate
                    if w > 0:           # попадалась разметка с нулевой шириной
                        letters.append((x, w, cv2.resize(letter_crop_resize(img=img, y=y, h=h, x=x, w=w),
                                                         (out_size, out_size), interpolation=cv2.INTER_AREA)))
                """

    letters.sort(key=lambda z: z[0], reverse=False)
    # blocks.sort(key=lambda z: z[0], reverse=False)
    return letters  #, blocks


def extract_chars(file_path: str) -> list:
    with open(file_path, 'r', encoding="utf-8") as f:
        # удаляем пробелы и все символы кроме букв из текста и разбиваем на символы
        text = f.read().strip()  # удаляем табы и лишние пробелы
        text = text.replace(' ', '')  # удаляем оставшиеся пробелы
        text = re.findall(PATTERN, text)  # оставляем только буквы и цифры
        print(f'Список букв: {text}')

    return list(text)


def create_dir(path: os.path) -> os.path:
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def convert_cv2pil(image: cv2) -> Image:
    # Т.к. OpenCV по-умолчанию использует формат BGR, перед сохранением преобразовываем его в стандартный RGB
    # Не используем т.к. и входящие и исходящие изображения - GRAY
    color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    # print(f"PIL.Image size: {pil_image.size}")
    return pil_image


if __name__ == '__main__':
    # TODO добавить "следящий индекс" - сохранять во время работы индекс обработанного файла и при следующем запуске
    # todo стартовать с последнего сохраненного.
    # TODO решить проблему с падением на определенном этапе из-за размеров изображения, которое пытаемся обработать
    if not os.path.isdir(WORK_DIR):
        print(f"Искомая папка отсутствует:\n{WORK_DIR}")
        raise FileNotFoundError

    # temp_dir = os.path.join(WORK_DIR, TEMP_DIR)
    create_dir(EXPORT_DIR)
    image_paths, image_names = get_files(WORK_DIR)
    for id_i, image in enumerate(image_paths[START_INDEX:END_INDEX]):
        if image[-5] == 'a': continue              # игнорируем картинки в нижнем регистре

        print(f'ImageID: {id_i}\nImagePath: {image}')
        letters = extract_letters(file_path=image)
        if letters == []: continue    # если получаем пустой список - то переходим к следующей картинке

        export_path = create_dir(os.path.join(EXPORT_DIR, f"{image_names[id_i]}"))  # создаем папку для вывода
        text_path = os.path.join(WORK_DIR, f'{image_names[id_i]}.gt.txt')
        print(f'TextPath: {text_path}')
        chars = extract_chars(file_path=text_path)

        print(f'Len of Letters: {len(letters)}, Len of Chars: {len(chars)}')
        # если удалось распознать не все или лишние буквы - снижаем ошибку из-за пропусков нормализуя их количество
        min_len = min(len(letters), len(chars))
        letters = letters[:min_len]
        chars = chars[:min_len]

        try:
            # Выгрузка изображений букв в отдельные папки с преобразованием в RGB вместо BGR, который использует cv2
            for id_l, letter in enumerate(letters):
                if chars[id_l] not in LABELS: break     # ради снижения ошибок и экономии, игнорируем знаки не из списка
                # image = convert_cv2pil(image=letter)  # Преобразуем BGR формат OpenCV в RGB перед сохранением

                cv2.imwrite(os.path.join(export_path, f'{image_names[id_i]}_{id_l}.jpg'), letter[2])
                # cv2.imwrite(os.path.join(export_path, f'{image_names[id_i]}_{id_l}_block.jpg'), blocks[id_l][2])

                with open(os.path.join(export_path, f'{image_names[id_i]}_{id_l}.txt'), 'w') as f:
                    f.write(chars[id_l])

            # Выгрузка букв в отдельные файлы. Лишний цикл не нужен, т.к. сохраняем через цикл с изображениями
            """
            for id_c, char in enumerate(chars):
                # не обязательно запускать два цикла, при условии что количество букв и ярлыков - равно
                with open(os.path.join(export_path, f'{image_names[id_i]}_{id_c}.txt'), 'w') as f:
                    f.write(char)
            """
        except Exception as e:
            print(e)
            continue
