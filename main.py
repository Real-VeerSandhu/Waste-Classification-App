from tensorflow import keras
from PIL import Image
import numpy as np

resnet50 = keras.models.load_model('models/resnet50_gar_TESTINGPU.h5')
labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

def pred_img(model, x):
    img1 = Image.open(x).convert(mode="RGB")
    img1 = img1.resize((256,256))
    array1 = np.array(img1.getdata())
    img_np_array = np.reshape(array1, (256,256,3))

    a = np.expand_dims(img_np_array, axis=0)
    print(model.predict(a), labels)
    return labels[np.argmax(model.predict(a))]

a = pred_img(resnet50, 'test_images/bottle.jpg')

print('Prediction ->', a)