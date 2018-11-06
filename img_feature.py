"""
Image feature pre-processing program:
extracting image feature of every image in the dataset with pre-trained VGG16 provided by Keras.
"""
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import h5py
import time
import os

# load pretrained model
model = VGG16(weights="imagenet", include_top=False)

if __name__ == '__main__':
    dataPath = os.getcwd()+"/InstagramImage/"
    outputname = "insta_imgFeat.h5"

    f_out = h5py.File(outputname, "a")
    mean_pixel = [103.939, 116.779, 123.68]

    img_list = os.listdir(dataPath)
    count = 0
    start = time.time()
    for item in img_list[:]:
        name = dataPath + item
        try:
            img = image.load_img(name, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = model.predict(x)
            f_out[item[:-4]] = feat
        except:
            file_error.write(item + '\n')
            continue
        count += 1
        if count % 10000 == 0:
            print(count)


