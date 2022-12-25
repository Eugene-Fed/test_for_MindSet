# -*- coding: utf-8 -*-
import cv2
import numpy as np

image_test = cv2.imread(r"D:\work\test_comp_vision\test_for_MindSet\task_3_color_normalization_example.jpg")
image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)


def normalize_it(x, min_fact, max_fact):
    # function to apply for pixels value with vectorization
    return (x - min_fact) / (max_fact - min_fact)


# TODO - дополнить возможностью загрузки изображений по ссылке и объекта PIL.Image
def normalize_img_color(image):
    """
    Input: image as numpy.array() object - cv2.image. Strongly recommended GRAY
    Output: image as numpy.array() object - cv2.image. Strongly recommended GRAY
    Gets image as numpy.array() from cv2.Image and normalize values of pixels to [0.0-1.0] format.
    That increase image contrast
    Подсказка от GPTchat о том, как нормализовать значения к нужному диапазону.
    (x - min) / (max - min)   # нормализуем изображение к диапазону [0.0-1.0]
    x * (max - min) + min                   # преобразование нормального изображения в требуемому диапазону
    """
    match image.dtype:
        case 'uint8':
            image = image.astype(np.float32)
            image /= 255.0
        case 'int8':
            # TODO - Оптимизировать и проверить!
            image = image.astype(np.uint8)
            image = image.astype(np.float32)
            image /= 255.0
        case 'float32':
            pass
    blur = cv2.GaussianBlur(image, (15, 15), 0)  # блерим для того, чтобы исключить шум

    norm_vec = np.vectorize(normalize_it)  # Векторизуем функцию для нормализации массива
    min_fact = blur.min()
    max_fact = blur.max()

    image = norm_vec(image, min_fact, max_fact)
    image *= 255
    #image += 127
    image = image.astype(np.uint8)

    return image


if __name__ == '__main__':
    img_test_min, img_test_max = image_test.min(), image_test.max()
    print(img_test_min, img_test_max)

    image_norm = normalize_img_color(image_test)
    img_norm_min, img_norm_max = image_norm.min(), image_norm.max()
    print(img_norm_min, img_norm_max)

    cv2.imshow('original', image_test)
    cv2.imshow('NORMALIZED', image_norm)

    cv2.waitKey(0)
