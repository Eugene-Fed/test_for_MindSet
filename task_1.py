from PIL import Image
import pytesseract
import os

DIR = 'pass_photos'                 # Path to the directory with photos
LANG = 'rus'                        # Language of recognizing text
PSM = 6                             # Page Segmentation Mode (--psm). Read more in Tesseract --help
OEM = 3                             # Engine Mode (--oem). Read more in Tesseract --help


def get_files(directory: str) -> list:
    """
    Collecting a list of photo file names in work directory
    :param directory: Path to the directory with photos
    :return: List of file names
    """
    names = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            names.append(os.path.join(directory, filename))

    return names


def recognize(img_path: str, lang: str, psm: int, oem: int) -> str:
    """
    Just recognizing text on selected files
    :param img_path: Path to image
    :param lang: Language of recognizing text
    :param psm: Page Segmentation Mode (--psm). Read more in Tesseract --help
    :param oem: Engine Mode (--oem). Read more in Tesseract --help
    :return: All recognized text from image
    """
    config = f'--oem {oem} --psm {psm}'
    text_data = f"========IMAGE '{img_path}' ========\n"           # Just for formatting
    text_data = text_data + pytesseract.image_to_string(Image.open(img_path), lang=lang, config=config) + "\n"
    print(text_data)
    return text_data


if __name__ == '__main__':
    text = ""
    images = get_files(directory=DIR)
    for img in images:
        text = text + recognize(img_path=img, lang=LANG, psm=PSM, oem=OEM)
    print(f"Count of recognized files = {len(images)}")
    with open("task_1.txt", 'w') as f:
        f.write(text)
