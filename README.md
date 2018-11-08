# MaCon

Implementation and dataset of AAAI 2019 paper:
**Hashtag Recommendation for Photo Sharing Services**

### Intro
MaCon is a deep neural network model used to recommend hashtags for posts in
photo sharing services. It takes image, text and user id as inputs and outputs
top K recommendations.

In this repository, we provide code implemented with Python Keras APIs
and our dataset crawled from Instagram. These materials are also used in experiments
reported in our AAAI paper.

Please cite our work if you're going to use our dataset:
[bib here]

### Details
- MaCon.py

The main program of our model. It contains data loading module, training module, as well as test module.
After all the data get prepared, you can directly execute this script.

- img_feature.py

Image feature extraction program. Image feature extraction is part of the data pre-processing.
This program process all the images in the dataset and extracts their feature vectors with pre-trained
VGG-16 network provided by Keras, which is trained on ImageNet.

The feature vectors are stored in a h5 file and key is the filename of each image file.

- selfDef.py

Some self-defined Keras layers and functions used in MaCon.

- data

All data needed in our model.

### Requirements

- Python 3.6
- TensorFlow >= 1.7
- Keras >= 2.1.5
- h5py >= 2.7.1

