# -*- coding: utf-8 -*-
import cv2
import numpy as np

image_test = cv2.imread(r"D:\work\test_comp_vision\test_for_MindSet\pass_temp\2_symbols.jpg")
image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)


def normalize_it(x, old_min, old_max, new_min, new_max):
    """
    :param x: the pixel value
    :param old_min: minimum pixels value of original image
    :param old_max: maximum pixels value of original image
    :param new_min: normalized minimum value
    :param new_max: normalized maximum value
    :return: normalized pixel value
    Moved the calculations into a separate function to apply it to np.array.
    return (x - min_fact) / (max_fact - min_fact)
    """
    if type(x) == 'int':                                      # if dtype is `int` then divide entirely
        x_norm = (x - old_min) * (new_max - new_min) // (old_max - old_min) + new_min
    else:                                           # else divide with preservation of the fractional part
        x_norm = (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

    x_norm = max(x_norm, new_min)           # if `x_norm` is oun of range [new_min .. new_max]
    x_norm = min(x_norm, new_max)           # then use `new_min` or `new_max` value

    return x_norm


# TODO - дополнить возможностью загрузки изображений по ссылке и объекта PIL.Image
def normalize_img_color(image, power='high'):
    """
    :param image: image as numpy.array() object - cv2.image. Strongly recommended GRAY
    :param power: power of image bluring for color normalization. 'low', 'norm', 'high'
    :return: image as numpy.array() object - cv2.image. Strongly recommended GRAY
    Gets image as numpy.array() from cv2.Image and normalize values to min-max range of `image` datatype
    That increase image contrast
    Hint from GPTchat on how to normalize values to the desired range.
    y = (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    x * (max - min) + min                   # reverse operation
    """
    image_dtype = image.dtype               # get original image np.dtype to use it later
    try:
        if image_dtype in ('uint8', 'int8'):
            new_min, new_max = np.iinfo(image_dtype).min, np.iinfo(image_dtype).max
        elif image_dtype in ('int16', 'int32', 'int64'):
            new_min, new_max = 0, 255
        elif image_dtype in ('uint16', 'uint32', 'uint64'):
            new_min, new_max = -128, 127
        elif image_dtype in ('float16', 'float32', 'float64', 'float128'):
            new_min, new_max = 0.0, 1.0
        else:
            image = image.astype(np.float32)
            new_min, new_max = 0.0, 1.0
    except Exception as e:
        print("Function require image as np.array with `int`, `uint` or `float` value.\n"
              "It can be obtained using the `OpenCV` or `PIL` library.")
        print(e)
        exit(1)

    # The more blur, the less noise in the image, which means that the contrast increases more
    if power == 'low':
        blur_kernel = (15, 15)
    elif power == 'norm':
        blur_kernel = (35, 35)
    else:
        blur_kernel = (55, 55)
    blur = cv2.GaussianBlur(image, blur_kernel, 0)  # get blur image to calc. `old_min` and `old_max` val without noise
    # cv2.imshow('BLURED', blur)
    # cv2.waitKey(0)

    old_min = blur.min()
    old_max = blur.max()

    norm_vec = np.vectorize(normalize_it)  # vectorize function to normalize all values of the np.array

    # return image after normalization with type as like as input image
    return norm_vec(image, old_min, old_max, new_min, new_max).astype(image_dtype)


if __name__ == '__main__':
    img_test_min, img_test_max = image_test.min(), image_test.max()
    image_norm_1 = normalize_img_color(image_test, power='low')
    image_norm_2 = normalize_img_color(image_test, power='norm')
    image_norm_3 = normalize_img_color(image_test, power='high')
    img_norm_min, img_norm_max = image_norm_3.min(), image_norm_3.max()

    print(image_test.dtype)
    print(img_test_min, img_test_max)
    print(image_norm_3.dtype)
    print(img_norm_min, img_norm_max)

    cv2.imshow('original', image_test)
    cv2.imshow('NORMALIZED', image_norm_1)
    cv2.imshow('NORMALIZED 2', image_norm_2)
    cv2.imshow('NORMALIZED 3', image_norm_3)

    cv2.waitKey(0)
