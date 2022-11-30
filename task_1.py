from doctr.models import ocr_predictor
from doctr.io import DocumentFile

IMAGES = ['pass_photos/0.jpeg', 'pass_photos/1.jpeg', 'pass_photos/2.jpeg', 'pass_photos/3.jpeg',
          'pass_photos/4.png', 'pass_photos/5.jpeg', 'pass_photos/6.jpeg', 'pass_photos/7.jpeg',
          'pass_photos/8.jpeg', 'pass_photos/9.jpeg']
model = ocr_predictor(det_arch='linknet_resnet18_rotation', reco_arch='master', pretrained=True)
# model = ocr_predictor(pretrained=True)

doc1 = DocumentFile.from_images(IMAGES[0])
docs = DocumentFile.from_images(IMAGES)
result = model(docs)
result.show(docs)
#print(result)


# pr.catching("pass_photos/0.jpeg")
# pr.download('pass_photos/4.png', '4.png')
