## test_for_MindSet
Test work for an internship at **MindSet**

### TASK 1
#### Since it was necessary to check the performance of several libraries, each of which pulls its own dependencies, it was easier for me to complete this task in PyCharm with a separate virtual environment for the project.  
To run **task_1.py** you should add photos to directory **pass_photos/**  
Requires library preinstallation **PIL** and **Tesseract-OSR** with Russian *"traineddata"*  
<https://tesseract-ocr.github.io/tessdoc/Installation.html>  
<https://github.com/tesseract-ocr/tessdata/blob/main/rus.traineddata>  

Text recognition is poor. But it turned out to be the easiest and most effective way of all that I tried before. Other libraries either do not recognize the Russian language at all, or had to face a huge number of problems during the installation process, or the volume of recognized text is even lower.

### TASK 2
I used the Yandex Cloud API
To execute this code, you need to follow the instructions for create Yandex.Cloud accaunt and obtaining the primary Yandex.ID token:  
<https://cloud.yandex.ru/docs/vision/operations/ocr/text-detection>  
Then it should work

### TASK 3
**WORK IN PROGRESS: ...95%...**  
You can use created *ru_emnist_letters_100k_b64_e60.h5* model fo recognition. This will allow you not to start generating the model in the "Обучаем модель на датасете" block. And in this case, you do not have to generate the dataset yourself.  
And you can try generate new model on ruEMNIST handwrited dataset from internet. May be it will work better.  
<https://www.kaggle.com/datasets/olgabelitskaya/handwritten-russian-letters>  
<https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset>  


## Additional files
### parse_ru-mnist.py  
Self maid script to parse *Russian News Corpus* to letter.jpg + letter.txt dataset  
<https://github.com/maxoodf/russian_news_corpus>  

### dataset_generator.ipynb  
It generates .IDX dataset from parsed *Russian News Corpus*  
