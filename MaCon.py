from keras.models import Model
from keras.layers import Input, Reshape, Dense, Embedding, Dropout, LSTM, Lambda, Concatenate, \
    Multiply, RepeatVector, Permute, Flatten, Activation
import keras.backend as K
from keras import optimizers

from selfDef import myLossFunc, Attention, coAttention_para, zero_padding, tagOffSet

import pickle
import h5py
import numpy as np

num_tags = 3896
num_words = 212000
index_from_text = 3
index_from_tag = 2
seq_length = 30    # max length of text sequence
batch_size = 512
embedding_size = 300
attention_size = 200
dim_k = 100
num_region = 7*7
drop_rate = 0.75
maxTagLen = 48    # max length of tag sequence
num_epoch = 30
numHist = 2    # historical posts number for each user
numTestInst = 64264    # if you're going to use predict_generator, modify this parameter as your testSet size.
testBatchSize = 40

print("Experiment parameters:")
print("embedding_size: %d, num_epoch: %d" % (embedding_size, num_epoch))

# load data
h5 = h5py.File("insta_imgFeat.h5", "r")
text_train, tag_train, ids_train, user_train = pickle.load(open("trainSet_20_%d.pkl"%numHist, "rb"))
user_post_samples = pickle.load(open("user_post_samples_%d.pkl"%numHist, "rb"))
text_train = zero_padding(text_train, seq_length)

text_test, tag_test, ids_test, user_test = pickle.load(open("testSet_20_%d.pkl"%numHist, "rb"))
text_test = zero_padding(text_test, seq_length)
tag_test = list(tag_test)
tmp_test_tag = []
for index in range(len(tag_test)-(numTestInst % testBatchSize)):
    tmpArray = np.zeros(num_tags)
    tmpArray[np.array(tag_test[index], dtype=np.int32)] = 1
    tmp_test_tag.append(tmpArray)
test_tag = np.array(tmp_test_tag)


# Loading all data into memory always causes OOM error. So we suggest keeping these two batchMakers.
def batchMaker(texts, tags, ids, users):
    shape = texts.shape[0]
    text_copy = texts.copy()
    tag_copy = tags.copy()
    ids_copy = ids.copy()
    users_copy = users.copy()

    indices = np.arange(shape)
    np.random.shuffle(indices)
    text_copy = list(text_copy[indices])
    tag_copy = list(tag_copy[indices])
    ids_copy = np.array(ids_copy)[indices]
    users_copy = users_copy[indices]

    i = 0
    while True:
        if i + batch_size <= shape:
            img_train = []
            tmp_train_text = []
            tmp_train_tag = []
            tmp_train_hist_text = []
            tmp_train_hist_img = []
            tmp_train_hist_tag = []

            for index in range(i, i+batch_size):
                data = h5.get(ids_copy[index])
                np_data = np.array(data)
                if np_data.shape != () and users_copy[index] in user_post_samples.keys():
                    img_train.append(np_data)
                    tmp_train_text.append(np.array(text_copy[index], dtype=np.int32))
                    tmpArray = np.zeros(num_tags)
                    tmpArray[np.array(tag_copy[index], dtype=np.int32)] = 1
                    tmp_train_tag.append(tmpArray)
                    tmp_img = []
                    tmp_text = []
                    tmp_tag = []
                    for inst in user_post_samples[users_copy[index]]:
                        tmp_img.append(np.array(h5.get(inst[2])))
                        tmp_text.append(np.array(inst[0]))
                        tmp_tag.append(np.array(tagOffSet(inst[1], index_from_tag)))
                    tmp_train_hist_text.append(np.array(zero_padding(tmp_text, seq_length)))
                    tmp_train_hist_img.append(np.array(tmp_img))
                    tmp_train_hist_tag.append(np.array(zero_padding(tmp_tag, maxTagLen)))
            text_train = np.array(tmp_train_text)
            tag_train = np.array(tmp_train_tag)
            img_train = np.squeeze(np.array(img_train))
            hist_text_train = np.array(tmp_train_hist_text)
            hist_img_train = np.squeeze(np.array(tmp_train_hist_img))
            hist_tag_train = np.array(tmp_train_hist_tag)

            yield [img_train, text_train,
                   hist_img_train[:, 0, :, :, :], hist_text_train[:, 0, :], hist_tag_train[:, 0, :],
                   hist_img_train[:, 1, :, :, :], hist_text_train[:, 1, :], hist_tag_train[:, 1, :]], tag_train
            i+=batch_size
        else:
            i= 0
            indices = np.arange(shape)
            np.random.shuffle(indices)
            text_copy = np.array(text_copy)
            tag_copy = np.array(tag_copy)
            text_copy = list(text_copy[indices])
            tag_copy = list(tag_copy[indices])
            ids_copy = np.array(ids_copy)[indices]
            users_copy = users_copy[indices]
            continue


def batchMakerTest(texts, tags, ids, users):
    shape = texts.shape[0]
    text_copy = texts.copy()
    tag_copy = tags.copy()
    ids_copy = ids.copy()
    users_copy = users.copy()

    i = 0
    while True:
        if i + testBatchSize <= shape:
            img_test = []
            tmp_test_text = []
            tmp_test_tag = []
            tmp_test_hist_text = []
            tmp_test_hist_img = []
            tmp_test_hist_tag = []

            for index in range(i, i + testBatchSize):
                data = h5.get(ids_copy[index])
                np_data = np.array(data)
                if np_data.shape != () and users_copy[index] in user_post_samples.keys():
                    img_test.append(np_data)
                    tmp_test_text.append(np.array(text_copy[index], dtype=np.int32))
                    tmpArray = np.zeros(num_tags)
                    tmpArray[np.array(tag_copy[index], dtype=np.int32)] = 1
                    tmp_test_tag.append(tmpArray)
                    tmp_img = []
                    tmp_text = []
                    tmp_tag = []
                    for inst in user_post_samples[users_copy[index]]:
                        tmp_img.append(np.array(h5.get(inst[2])))
                        tmp_text.append(inst[0])
                        tmp_tag.append(tagOffSet(inst[1], index_from_tag))
                    tmp_test_hist_text.append(np.array(zero_padding(tmp_text, seq_length)))
                    tmp_test_hist_img.append(np.array(tmp_img))
                    tmp_test_hist_tag.append(np.array(zero_padding(tmp_tag, maxTagLen)))
            text_test = np.array(tmp_test_text)
            tag_test = np.array(tmp_test_tag)
            img_test = np.squeeze(np.array(img_test))
            hist_text_test = np.array(tmp_test_hist_text)
            hist_img_test = np.squeeze(np.array(tmp_test_hist_img))
            hist_tag_test = np.array(tmp_test_hist_tag)
            yield [img_test, text_test, hist_img_test[:, 0, :, :, :], hist_text_test[:, 0, :], hist_tag_test[:, 0, :],
                   hist_img_test[:, 1, :, :, :], hist_text_test[:, 1, :], hist_tag_test[:, 1, :]], tag_test
            i += testBatchSize
        else:
            i = 0
            continue


def modelDef():
    inputs_img = Input(shape=(7, 7, 512))
    inputs_text = Input(shape=(seq_length,))
    inputs_hist_img_0 = Input(shape=(7, 7, 512))
    inputs_hist_img_1 = Input(shape=(7, 7, 512))
    inputs_hist_text_0 = Input(shape=(seq_length,))
    inputs_hist_text_1 = Input(shape=(seq_length,))
    inputs_hist_tag_0 = Input(shape=(maxTagLen,))
    inputs_hist_tag_1 = Input(shape=(maxTagLen,))

    # shared layers
    tagEmbeddings = Embedding(input_dim=num_words + index_from_tag, output_dim=embedding_size,
                              mask_zero=True, input_length=maxTagLen)
    textEmbeddings = Embedding(input_dim=num_words + index_from_text, output_dim=embedding_size,
                               mask_zero=True, input_length=seq_length)
    lstm = LSTM(units=embedding_size, return_sequences=True)
    dense = Dense(embedding_size, activation="tanh", use_bias=False)
    reshape = Reshape(target_shape=(num_region, 512))
    coAtt_layer = coAttention_para(dim_k=dim_k)
    tag_att = Attention(attention_size)

    # query post representation
    text_embeddings = textEmbeddings(inputs_text)
    tFeature = lstm(text_embeddings)
    iFeature = reshape(inputs_img)
    iFeature = dense(iFeature)
    co_feature = coAtt_layer([tFeature, iFeature])

    # historical posts representation
    hist_tag_0 = tag_att(tagEmbeddings(inputs_hist_tag_0))
    hist_text_0 = textEmbeddings(inputs_hist_text_0)
    hist_tfeature_0 = lstm(hist_text_0)
    hist_ifeature_0 = reshape(inputs_hist_img_0)
    hist_ifeature_0 = dense(hist_ifeature_0)
    hist_cofeature_0 = coAtt_layer([hist_tfeature_0, hist_ifeature_0])

    hist_tag_1 = tag_att(tagEmbeddings(inputs_hist_tag_1))
    hist_text_1 = textEmbeddings(inputs_hist_text_1)
    hist_tfeature_1 = lstm(hist_text_1)
    hist_ifeature_1 = reshape(inputs_hist_img_1)
    hist_ifeature_1 = dense(hist_ifeature_1)
    hist_cofeature_1 = coAtt_layer([hist_tfeature_1, hist_ifeature_1])

    sim_0 = Multiply()([hist_cofeature_0, co_feature])
    sim_0 = RepeatVector(1)(Concatenate()([sim_0, hist_tag_0]))
    sim_1 = Multiply()([hist_cofeature_1, co_feature])
    sim_1 = RepeatVector(1)(Concatenate()([sim_1, hist_tag_1]))
    sims = Concatenate(axis=1)([sim_0, sim_1])

    attention = Dense(1, activation='tanh')(sims)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(2 * embedding_size)(attention)
    attention = Permute([2, 1])(attention)

    influence = Multiply()([sims, attention])
    influence = Lambda(lambda x: K.sum(x, axis=1))(influence)
    influence = Dense(embedding_size)(influence)

    h = Concatenate()([co_feature, influence])
    dropout = Dropout(drop_rate)(h)
    Softmax = Dense(num_tags, activation="softmax", use_bias=True)(dropout)
    model = Model(inputs=[inputs_img, inputs_text, inputs_hist_img_0, inputs_hist_text_0, inputs_hist_tag_0,
                          inputs_hist_img_1, inputs_hist_text_1, inputs_hist_tag_1],
                  outputs=[Softmax])

    model.compile(optimizer="adam", loss=myLossFunc)
    return model


def evaluator(y_true, y_pred, top_K):
    acc_count = 0
    precision_K = []
    recall_K = []
    f1_K = []

    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[-top_K:]
        if np.sum(y_true[i, top_indices]) >= 1:
            acc_count += 1
        p = np.sum(y_true[i, top_indices]) / top_K
        r = np.sum(y_true[i, top_indices]) / np.sum(y_true[i, :])
        precision_K.append(p)
        recall_K.append(r)
        if p != 0 or r != 0:
            f1_K.append(2 * p * r / (p + r))
        else:
            f1_K.append(0)
    acc_K = acc_count * 1.0 / y_pred.shape[0]

    return acc_K, np.mean(np.array(precision_K)), np.mean(np.array(recall_K)), np.mean(np.array(f1_K))


if __name__ == "__main__":
    for top_K in [7]:
        model = modelDef()

        F = 0.0
        res_file = open("record_userHist_%d_%d.txt"%(embedding_size, numHist), "a")
        string = "Embedding_size = %d \t Top- %d\n" % (embedding_size, top_K)
        res_file.write(string)
        print("Start Training...")
        for epoch in range(num_epoch):
            history = model.fit_generator(
                generator=batchMaker(text_train, tag_train, ids_train, user_train),
                steps_per_epoch=int(text_train.shape[0] / batch_size),
                epochs=1,
                verbose=1,)
            y_pred = model.predict_generator(generator=batchMakerTest(text_test, tag_test, ids_test, user_test),
                                             steps=int(numTestInst / testBatchSize),
                                             verbose=1)
            acc, precision, recall, f1 = evaluator(test_tag, y_pred, top_K)

            print("Top %d, Epoch: %d,accuracy: %.6f, precision: %.6f, recall: %.6f, f1: %.6f" %
                  (top_K, epoch, acc, float(precision), float(recall), float(f1)))
            if f1 >= F:
                model.save_weights("model_best_%d_%d.h5"%(embedding_size, numHist))
                res_file = open("record_userHist_%d_%d.txt"%(embedding_size, numHist), "a")
                string = "Epoch: %d,accuracy: %.6f, precision: %.6f, recall: %.6f, f1: %.6f \n" % (
                epoch, acc, float(precision), float(recall), float(f1))
                res_file.write(string)
                res_file.close()
                F = f1
    print("Training Process Completed.")