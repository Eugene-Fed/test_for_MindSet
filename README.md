## test_for_MindSet
Test work for an internship at **MindSet**

### TASK 1: Select open libraries for document recognition and conduct passport recognition.  
**Since it was necessary to check the performance of several libraries, each of which pulls its own dependencies, it was easier for me to complete this task in PyCharm with a separate virtual environment for the project.**  
To run `task_1.py` you should add photos to directory `pass_photos/`  
Requires library preinstallation **PIL** and **Tesseract-OSR** with Russian *"traineddata"*  
<https://tesseract-ocr.github.io/tessdoc/Installation.html>  
<https://github.com/tesseract-ocr/tessdata/blob/main/rus.traineddata>  

Text recognition is poor. But it turned out to be the easiest and most effective way of all that I tried before. Other libraries either do not recognize the Russian language at all, or had to face a huge number of problems during the installation process, or the volume of recognized text is even lower.

### TASK 2: Select APIs with which you can perform recognition, perform recognition.  
I used the Yandex Cloud API
To execute this code, you need to follow the instructions for create Yandex.Cloud accaunt and obtaining the primary Yandex.ID token:  
<https://cloud.yandex.ru/docs/vision/operations/ocr/text-detection>  
Then it should work

### TASK 3: Train your own algorithm (pytorch, tensorflow) that recognizes the full name from the passport (not using ready-made OCR libraries, but using open text recognition data).  
**WORK IN PROGRESS: ...99%...**  
1. Add Passport photos into `pass_photos/` folder  
2. Open `task_3.ipynb`. If you wanna use your EMNIST-like dataset then change path of `DATASET_IMG` and `DATASET_CLS` variables.  
3. Execute all blocs from the first to part `Готовим модель и train/test`. If you wanna use already created Model - go to **Step 5.**  
4. Execute all blocs from part `Готовим модель и train/test` to `Обучаем модель на датасете`. Then specify size of **Batch** and count of **Epochs** (*watch "Ideas for TASK 3"*). Run `Обучаем модель на датасете` block. Model with current settings will be calculating about 4.5 hours.  
5. Execute all blocs from `Детекция данных из паспорта и сохранение фоток букв в файлы` to the End. Result will be printed and saved to the `task_3.txt` file.  

If you pass **Step 4** then it will be used already created `ru_emnist_letters_100k_b64_e150_upper2.h5` model for recognition. This will allow you not to start generating the model in the `Обучаем модель на датасете` block. And in this case, you do not have to generate the dataset yourself.  
And you can try generate new model on ruEMNIST handwrited dataset from internet. May be it will work better.  
<https://www.kaggle.com/datasets/olgabelitskaya/handwritten-russian-letters>  
<https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset>  
Now the model correctly recognizes about 50% of characters. There are several ideas for improving results.  

### Ideas for TASK 3:  
The best results of training with current parameters is: **loss ~ 0.0136**, **accuracy: ~ 0.9997**.  
The best practice is training model only on uppercase letters. It gets perfect result on validate data, but worse on data from passport. I should work with passport's photo to do it more contrast.  
There is no need to generate more than **55 epochs** with size of **64 batch**, because the performance is no longer improving. You should try changing other settings.
- [x] Hide *X_train*, *X_test* normalization in block `Загружаем датасет Часть 2 / 2`.  
After 103 epochs - loss: ~ 1.3, accuracy: ~ 0.78
- [x] Convert all letters in the training dataset to uppercase and thus reduce the classifier from 76 to 41 (without '0' and '3' numbers).  
After 150 epochs - loss: ~ 0.98, accuracy: ~ 0.78
- [x] Rewrite `dataset_generator.ipynb` for use only uppercase letters to create dataset. Original dataset has 'a' and 'b' liters in it names that means: 'a' - lowercase, 'b' - uppercase phrases.  
After 150 epochs - loss: ~ 0.51, accuracy: ~ 0.91
- [x] Add margins around characters in production data and increase image contrast (make the background lighter). As an example, take images from the training sample.
- [x] Add and use `Tesseract OCR` library to check the quality of images used for recognition.
- [x] ~~Add train data augmentation for Model creation.~~ Add image normalization layer into the Model for increase accuracy of training and recognition.
After 150 epochs - loss: ~ 0.33, accuracy: ~ 0.94
- [x] Add Train, Validation and Test datasets of 125k, 25k and 25k elements respectively.
- [x] ~~Increase resolution of train and production images from 28 to 32 pixels.~~ Original data has resolution of letters less then 28 pixels.
- [x] Add image normalization before searching symbol blocks.
- [x] Remove numbers from the training dataset and leave only 33 uppercase letters. We can check if a character matches a pattern before adding a new element to the dataset.
- [x] ~~Check dataset for errors - **WORK IN PROGRESS: ...10%...**~~
- [x] Rewrite `parse_ru-mnist.py` to minimize errors in image-labels data.  
**After 150 epochs - loss: ~ 0.0136, accuracy: ~ 0.9997**
- [x] ~~Rewrite `normalize_color` function in `task_3.ipynb` to detect symbols without *blur* and *cv2.erode()*~~ I can't do that because searching of word's blocks need to blur and erode image. I tried to use binarization threshhold for symbols searching, but it doesn't work.
- [x] Understand how image value normalization works by dividing its values by 255.
- [x] Create Image color normalization module `color_normalization.py`.
- [x] ~~Find the reason why when you try to normalize images in the emnist parser, images are saved in color inversion.~~  
I rewrote `color_normalization.py` so that it returns the image in the same format as it receives it.
- [ ] ~~Delete symbols' scale before detection, because this oparation degrades the quality.~~ Check all methods of image scaling.  
`passport_data_parser()` line `40`
- [ ] Try to erode image before symbol detection.
- [ ] Get train dataset without white margins around symbols. It should be filled background color, not clean white. May be I should rewrite `parse_ru-mnist.py` to expand narrow symbol to square instead of padding the width with white margins.
- [ ] Train model on more font variants.
- [ ] Create adaptive setting of Brihtness/Contrast.
- [ ] Edit `dataset_generator` to save pictures in **int8** representation because we use value of pixels brightness beth 0..255.

## Additional files
### parse_ru-mnist.py  
Self maid script to parse *Russian News Corpus* `phrase.jpg + phrase.txt` to `letter.jpg + letter.txt` dataset  
<https://github.com/maxoodf/russian_news_corpus>  

### dataset_generator.ipynb  
It generates .IDX dataset from parsed *Russian News Corpus*  
**IMPORTANT**  
I was not able to make a dataset of more than **150,000** image options, because it can generate `Memory Error` occurs when saving the finished dataset. I am using *Intel Core i7* with *16GB of memory*. Perhaps it will be possible to make a bigger dataset with `int8` np.dtype for image representation.

### RECOMENDATION FOR FUTURE:  
1. **CHECK YOU DATASET**!!! It can has several errors, that will give you **big shit**.  
2. **NORMALIZE YOU IMAGES**!!! This increases the contrast and the ability to recognize elements in the image.
