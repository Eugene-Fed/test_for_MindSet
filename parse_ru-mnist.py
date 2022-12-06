import cv2
import numpy as np
import os
import math

# Список всех настроечных параметров/констант
WORK_DIR = r"D:\work\test_comp_vision\datasets\!_lines_w25"
TEMP_DIR = r"parsed"


# Функция для получения списка файлов из каталога с фотографиями (как в task_1 и task_2)
def get_files(directory: str) -> list:
    paths = []              # полный пути к файлам изображений
    names = []              # имя без расширения
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            paths.append(os.path.join(directory, filename))
            names.append(filename.split('.')[0])

    return paths, names

#
# def resize_image(image, out_size: int):
#     h, w = image.shape[0], image.shape[1]
#     size_max = max(w, h)
#     img_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
#     if w > h:
#         # Изображение широкое
#         y_pos = size_max // 2 - h // 2
#         img_square[y_pos:y_pos + h, 0:w] = image
#     elif w < h:
#         # Изображение узкое
#         x_pos = size_max // 2 - w // 2
#         img_square[0:h, x_pos:x_pos + w] = image
#     else:
#         img_square = image
#
#     return cv2.resize(img_square, (out_size, out_size))


def letter_crop_resize(img, y, h, x, w):
    letter_crop = img[y:y + h, x:x + w]

    # Resize letter canvas to square
    size_max = max(w, h)
    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
    if w > h:
        y_pos = size_max // 2 - h // 2
        letter_square[y_pos:y_pos + h, 0:w] = letter_crop
    elif w < h:
        x_pos = size_max // 2 - w // 2
        letter_square[0:h, x_pos:x_pos + w] = letter_crop
    else:
        letter_square = letter_crop

    return letter_square


def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((2, 2), 'uint8')
    erosion = cv2.erode(gray, kernel, iterations=1)
    thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = 255 - thresh

    result = invert

    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output = img.copy()
    cv2.waitKey(0)

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
        limit = 10          # ограничение на мелкие символы, чтобы их не вырезать
        if limit < h < img.shape[0] and limit < w < img.shape[1]:  # игнорируем маленькие блоки, а также блок размером с изображение
            c = 1.25  # просто коэффициент, рассчитанный на широкие буквы вроде Ж, М, Ш и т.д., чтобы их не делило
            if w < h * c:
                letters.append((x, w, cv2.resize(letter_crop_resize(gray, y=y, h=h, x=x, w=w),
                                                 (out_size, out_size), interpolation=cv2.INTER_AREA)))
            else:
                symbol_count = math.ceil(w / h)  # округляем символы до большего целого
                w = math.floor(w / symbol_count)  # округляем ширину в пикселях до меньшего целого
                for i in range(symbol_count):
                    x += i * w
                    # Resize letter to 28x28 and add letter and its X-coordinate
                    letters.append((x, w, cv2.resize(letter_crop_resize(gray, y=y, h=h, x=x, w=w),
                                                     (out_size, out_size), interpolation=cv2.INTER_AREA)))

    letters.sort(key=lambda x: x[0], reverse=False)
    return letters


def create_dir(path: os.path) -> os.path:
    if not os.path.exists(path):
        os.mkdir(path)
    return path


if __name__ == '__main__':
    temp_dir = os.path.join(WORK_DIR, TEMP_DIR)
    create_dir(temp_dir)
    image_paths, image_names = get_files(WORK_DIR)
    for id_i, image in enumerate(image_paths[:]):
        print(f'ImageID: {id_i}\nImagePath: {image}')
        path = create_dir(os.path.join(temp_dir, f"{image_names[id_i]}"))
        letters = letters_extract(image_file=image)
        print(f'Len of Letters: {len(letters)}')
        for id_l, letter in enumerate(letters):
            cv2.imwrite(os.path.join(path, f'{image_names[id_i]}_{id_l}.jpg'), letter[2])


