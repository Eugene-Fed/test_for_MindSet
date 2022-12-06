import cv2
import numpy as np
import os
import math

# Список всех настроечных параметров/констант
WORK_DIR = r"D:\work\test_comp_vision\datasets\!_lines_w25_test"
TEMP_DIR = r"parsed"
# TEST_FILE = 'pass_photos/1.jpeg'
IMG_HEIGHT = 1000            # требуемый размер фото для нормализации всех изображений
IMG_WIDTH = 600              # т.к. в задачу входит прочитать только ФИО, обрезаю серию/номер чтобы не усложнять распознавание
INDENT_LEFT = 220            # обрезаем фото т.к. без него получается лучше разделить фото на куски текста
INDENT_TOP = 40              # обрезаем лишнюю часть паспорта снизу
INDENT_BOTTOM = 120          # обрезаем нижние поля
SCALE_FACTOR = 8             # во сколько раз увеличиваем вырезанные слова для дальнейшей обработки букв
SYMBOL_SYZE = 28             # размеры выходного датасета


# Функция для получения списка файлов из каталога с фотографиями (как в task_1 и task_2)
def get_files(directory: str) -> list:
    paths = []              # полный пути к файлам изображений
    names = []              # имя без расширения
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            paths.append(os.path.join(directory, filename))
            names.append(filename.split('.')[0])

    return paths, names


# Изменение размера
def scale_image(image, scale):  # принимаем объект изображения OpenCV

    # получаем текущий размер, вычисляем искомый и создаем измененное изображение
    height, width = image.shape[0], image.shape[1]
    img_width = int(width * scale)
    img_height = int(height * scale)
    img = cv2.resize(image, (img_width, img_height))
    # img = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def resize_image(image, out_size: int):
    h, w = image.shape[0], image.shape[1]
    size_max = max(w, h)
    img_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
    if w > h:
        # Изображение широкое
        y_pos = size_max // 2 - h // 2
        img_square[y_pos:y_pos + h, 0:w] = image
    elif w < h:
        # Изображение узкое
        x_pos = size_max // 2 - w // 2
        img_square[0:h, x_pos:x_pos + w] = image
    else:
        img_square = image

    # Resize letter to 28x28 and add letter and its X-coordinate
    # result = (x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA))
    # return cv2.resize(img_square, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(img_square, (out_size, out_size))


# Нормализация размеров и вырезка нужной части паспорта для обработки
def normalize_size(image):  # принимаем объект изображения OpenCV

    # нормализуем фото к нужному размеру
    old_height = image.shape[0]  # получаем исходную высоту
    resize_scale = IMG_HEIGHT / old_height  # считаем коэффициент масштабирования изображения до требуемого
    img = scale_image(image=image, scale=resize_scale)
    new_width = img.shape[1]  # получаем новую ширину

    # обрезаем паспорт до страницы с фото
    x0 = INDENT_LEFT  # отступ слева, т.к. корочка и фото нам не важны
    y0 = IMG_HEIGHT // 2 + INDENT_TOP  # обрезка сверху, т.к. верхняя страница с местом выдачи нам не важна
    x1 = new_width if new_width < IMG_WIDTH else IMG_WIDTH  # обрезаем все лишнее справа, если есть разворот с пропиской
    y1 = IMG_HEIGHT - INDENT_BOTTOM
    img = img[y0:y1, x0:x1]  # обресанный кусок изображения

    return img


# Подготовка изображений для распознавания текста
def normalize_color(image):  # принимаем объект изображения OpenCV

    # обесцвечиваем и пытаемся снизить шум с помощью размытия
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # преобразуем в ЧБ
    else:
        gray = image
    # blur = cv2.GaussianBlur(gray, (5,5), 0)         # коэффициент размытия подобран вручную

    # одно изображение используем для распознавания блоков текста. очередность преобраозвания найдена методом тыка
    kernel = np.ones((2, 2), 'uint8')
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_block = cv2.erode(gray, kernel, iterations=1)
    # img_block = cv2.dilate(img_block, kernel, iterations=1)
    _, img_block = cv2.threshold(img_block, 0, 255, cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV)
    img_block = cv2.morphologyEx(img_block, cv2.MORPH_OPEN, kernel, iterations=1)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Grayscale, Gaussian blur, Otsu's threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    erosion = cv2.erode(gray, kernel, iterations=1)
    dilation = cv2.dilate(gray, kernel, iterations=1)
    invert = 255 - closing

    # Повышение контраста
    if len(image.shape) > 2:
        imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        imghsv[:, :, 2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in
                           imghsv[:, :, 2]]
        contrast = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
        gray_contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)  # преобразуем в ЧБ

    # при коэффициенте 3 - лучше распознается Васлевский, при 5 - Соколов и Юмакаева
    img_symbol = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO + cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_block, gray  # Возвращаем контрастную картинку с разбивкой на блоки и простое ЧБ изображение


# Выделяем элементы текста из изображения
def search_blocks(image, sort_by: str, limit=0, sort_reverse=False):
    height, width = image.shape[0], image.shape[1]
    # получаем контуры больших пятен на изображении, внутри которых спрятан текст
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # output = image.copy()      #TODO - можно удалить

    print(f'Count of Block counoturs: {len(contours)}')
    blocks = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", x, y, w, h, hierarchy[0][idx])
        # hierarchy[i][0]: следующий контур текущего уровня
        # hierarchy[i][1]: предыдущий контур текущего уровня
        # hierarchy[i][2]: первый вложенный элемент
        # hierarchy[i][3]: родительский элемент
        # if hierarchy[0][idx][3] == 0:               # если элемент не является самым крупным
        cv2.rectangle(image, (x, y), (x + w, y + h), (70, 0, 0), 1)  # для отображаемой картинки
        if limit < h < height and limit < w < width:  # игнорируем маленькие блоки, а также блок размером с изображение
            block = image[y:y + h, x:x + w]  # вырезаем найденный блок из изображения
            # blocks.append((idx, y, h, x, w, block))
            # сохраняем габариты и изображение блока в список блоков. Загоняем в словарь, чтобы проще сортировать
            blocks.append({'idx': idx, 'y': y, 'h': h, 'x': x, 'w': w, 'block': block})

    # сортируем по нужному ключу: 'y' для вертикали или 'x' по горизонтали. так же можно и по индексу или размерам
    blocks.sort(key=lambda x: x.get(sort_by), reverse=sort_reverse)
    # print(blocks)
    return blocks
    # return image


# Режем блок изображения арифметически, если он шире одной высоты символа
def cut_blocks(image):
    height, width = image.shape[0], image.shape[1]
    C = 1.2  # просто коэффициент, рассчитанный на широкие буквы вроде Ж, М, Ш и т.д., чтобы их не делило
    if width < height * C:
        print(f'One symbol is True')
        return [image]
    else:
        print(f'One symbol is FALSE')
        result = []
        y, h, = 0, height  # высота и верхняя точке среза - всегда неизменны
        symbol_count = math.ceil(width / height)  # округляем символы до большего целого
        symbol_width = math.floor(width / symbol_count)  # округляем ширину в пикселях до меньшего целого

        # while image.shape[1] > image.shape[0]*C :
        for i in range(symbol_count):
            # symbol_count = round(image.shape[1] / image.shape[0])     # ширину делим на высоту ~ количество символов в блоке
            # symbol_count = math.ceil(image.shape[1] / image.shape[0])
            x = i * symbol_width
            print(f'y = {y}, h = {h}, x = {x}, symbol_width = {x + symbol_width}, width = {width}')
            result.append(image[y:h, x:x + symbol_width])
            print(f'symbol {i} is:\n{result[i]}')

        print(f'count of separeted symbols: {len(result)}')
        return result


def parse_images(work_dir: str, temp_dir: str):
    # Запускаем цикл по всем фото в рабочей папке
    image_paths, image_names = get_files(work_dir)
    print(f'Image_paths: {image_paths},\nImage_names: {image_names}')
    for id_i, image_path in enumerate(image_paths):  # идем по списку путей к изображениям

        # TODO - вместо второго списка имен сделать функцию, которая через RegExp будет это имя добывать
        result_dir = os.path.join(temp_dir, str(image_names[id_i]), '\\')

        image = cv2.imread(image_path)
        img_blocks, img_gray = normalize_color(image=image)   # получаем два изображения для разбивки на буквы
        cv2.imwrite(f'{temp_dir}/{image_names[id_i]}_blocs.jpg', img_blocks)
        cv2.imwrite(f'{temp_dir}/{image_names[id_i]}_symbols.jpg', image)

        symbols = search_blocks(image=img_blocks, sort_by='x')  # сортируем список слева направо

        try:
            for id_s, symbol in enumerate(symbols):  # можно забираем только первые 3 слова ФИО
                # из словаря обнаруженного блока текста забираем координаты и размер блока
                y, h, x, w = symbol['y'], symbol['h'], symbol['x'], symbol['w']
                img_symbol = img_gray[y:y + h, x:x + w]  # вырезаем слово по его координатам
                # img_symbol = image[y:y + h, x:x + w]     # вариант с повышением контраста
                img_symbol = resize_image(img_symbol, 28)  # увеличиваем изображение
                cv2.imwrite(os.path.join(temp_dir, f'{id_s}.jpg'), img_symbol)  # сохраняем файлы только для контроля

                # Доп. проверка на случай, если буквы плохо отделились
                for id_o, one_symbol in enumerate(cut_blocks(img_symbol)):
                    # print(f'element: {e}, one_symbol: {one_symbol}')
                    cv2.imwrite(os.path.join(temp_dir, f'{id_s}-{id_o}.jpg'), one_symbol)

        except IndexError:  # хотя сейчас Питон не выдает исключение, даже если в списке всего 1 элемент
            print("В паспорте найдено недостаточно данных. Попробуйте сделать более качественный скан.")


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
    # cv2.imshow("Erosion", erosion)
    thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # cv2.imshow("thresh", thresh)
    invert = 255 - thresh
    # cv2.imshow("Invert", invert)

    result = invert

    # Get contours
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

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    # cv2.imshow("Input", img)
    # cv2.waitKey(0)
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)
    # for i, letter in enumerate(letters[:5]):
    #     cv2.imshow(f"Letter {i}", letters[i][2])
    # cv2.waitKey(0)

    return letters


def img_to_str(model, image_file: str):
    letters = letters_extract(image_file)
    s_out = ""
    # for i in range(len(letters)):
    #     dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
    #     s_out += emnist_predict_img(model, letters[i][2])
    #     if (dn > letters[i][1]/4):
    #         s_out += ' '
    # return s_out


if __name__ == '__main__':
    temp_dir = os.path.join(WORK_DIR, TEMP_DIR)
    if not os.path.exists(temp_dir):
        print('Making dir')
        os.mkdir(temp_dir)
    # parse_images(work_dir=WORK_DIR, temp_dir=temp_dir)
    image_paths, image_names = get_files(WORK_DIR)
    for id_i, image in enumerate(image_paths[:10]):
        print(f'ImageID: {id_i}\nImagePath: {image}')
        letters = letters_extract(image_file=image)
        print(f'Len of Letters: {len(letters)}')
        for id_l, letter in enumerate(letters):
            print(f'LetterID: {id_l}')
            cv2.imwrite(f'{temp_dir}/{image_names[id_i]}_{id_l}.jpg', letter[2])


