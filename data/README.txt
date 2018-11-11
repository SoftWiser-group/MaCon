# Instagram.data
Every line in Instagram.data represents a post.
Every post has 4 attributes: post_id, words, hashtags, user_id. These attributes are organised in order and seperated by a tab.
For post with more than one words or hashtags, words or hashtags are seperated by a space.
You can download it here: https://pan.baidu.com/s/1i25OltzAmxbT4L8yvxAK0g retrieve code：41i6 

# vocabulary
Every line represents a word, consisting of word index and the word which are seperated by a tab.
Words are ordered based on frequency of occurence. 
You can use word id provided in this file and Instagram.data to acquire word sequence of text data.

# tags
Similar with vocabulary.

# InstagramImage.zip
It includes all images responding to the posts in the dataset. Every jpg file is named with the post_id of corresponding post. 
This file is about 40G and we put it on Baidu Cloud disk. 
You can download it here: https://pan.baidu.com/s/1Wh3gVwGPYWMqVMEWJEaPhw  retrieve code：rj5v 


# dataSplit.py
This program will produce train set, test set and history samples for each user.
