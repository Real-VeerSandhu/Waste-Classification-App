import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras

labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
resnet50 = keras.models.load_model('models/resnet50_gar_TESTINGPU.h5')

def main():

    def pred_img(model, x):
        img1 = Image.open(x).convert(mode="RGB")
        img1 = img1.resize((256,256))
        array1 = np.array(img1.getdata())
        img_np_array = np.reshape(array1, (256,256,3))

        a = np.expand_dims(img_np_array, axis=0)
        outputs = [labels[np.argmax(model.predict(a))], model.predict(a)]
        return outputs

    a = st.file_uploader('File uploader', type=['png', 'jpg'])

    print(type(a))

    if (type(a) != type(None)):
        image = Image.open(a)
        st.image(image, caption='Uploaded Image.')
        out = pred_img(resnet50, a)
        st.write(a.name + ' -> ', out[0], out[1])

if __name__ == '__main__':
    main()
# from joblib import load


# def main(): 

#     def load_model(file):
#         loaded_model = load(file)
#         return loaded_model

#     st.write("""
#     ## Testing Streamlit (checking for change)
#     """)
#     rf_env = load_model('models/crop_outlook_rfg1.joblib')

#     def env_rating(temp, soil_mois, hum):
#         result = rf_env.predict([[temp, soil_mois, hum]])
#         # print('Environment Rating: ' + str(round(result[0], 2)) + '%')
#         return str(round(result[0], 2))

#     temp2 = st.number_input('T ', 0, 100)
#     soil_mois2 = st.number_input('S', 0, 100)
#     hum2 = st.number_input('H ', 0, 100)

#     if st.button('Run Model'):
#         rating = env_rating(temp2, soil_mois2, hum2)
#         print('Rating:', rating)
#         st.write(str(rating) + '%')

# if __name__ == '__main__':
# 	main()