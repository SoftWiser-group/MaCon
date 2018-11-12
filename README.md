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

### Usage

- Data split:

In this step, we will divide the data into train set, test set and user history sample set.
Make sure that dataSplit.py and Instagram.data are under same dictionary.
Execute the python script:
```
python dataSplit.py
```
- Image feature extraction:

We extract feature vector of all images in this dataset with img_feature.py.
Download the zip files and unpack them. Then execute:
```
python img_feature.py
```
The feature vectors are stored in a h5 file which can be indexed with filename.

- Training and evaluation

When data preparations are done, run Macon.py:
```
python MaCon.py
```
Evaluation metrics in training process are recorded in record_userHist_EMBED_NUMHIST.txt
And model is stored in model_best_EMBED_NUMHIST.h5


### Requirements

- Python 3.6
- TensorFlow >= 1.7
- Keras >= 2.1.5
- h5py >= 2.7.1

### Citing
**If you find our work or dataset useful in your research, please cite the following paper:**
```
@inproceedings{zhangMaCon2018,
 author = {Suwei Zhang, Yuan Yao, Feng Xu, Hanghang Tong, Xiaohui Yan, Jian Lu},
 title = {Hashtag Recommendation for Photo Sharing Services},
 booktitle = {33rd AAAI Conference on Artificial Intelligence},
 year = {2019},
}
```

