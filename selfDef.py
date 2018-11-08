"""
some DIY components used in MaCon
    Attention -- DIY attention layer. Typical attention definition.
    coAttention_para: DIY parallel co-attention layer.
    myLossFunc -- DIY loss function.
    tagOffSet -- Used to pre-process tag sequence of historical posts.
    zero_padding -- Used to pre-process tag or text sequence: when len<max_length, fill zeros after the sequence;
                                                              when len>=max_length, reserve the subsequence of max_length
"""
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from keras import activations, initializers
import numpy as np

REMOVE_FACTOR = -10000


class Attention(Layer):
    """
    self defined text attention layer.
    input: hidden text feature
    output: summarized text feature with attention mechanism

    input shape: (batch_size, seq_length, embedding_size)
    output shape: (batch_size, embedding_size)
    """
    def __init__(self, units, return_alphas=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True
        self.return_alphas = return_alphas

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Create a trainable weight variable for this layer.
        self.w_omega = self.add_weight(name='w_omega',
                                       shape=(input_dim, self.units),
                                       initializer='random_normal',
                                       trainable=True)
        self.b_omega = self.add_weight(name='b_omega',
                                       shape=(self.units,),
                                       initializer='zeros',
                                       trainable=True)
        self.u_omega = self.add_weight(name='u_omega',
                                       shape=(self.units,),
                                       initializer='random_normal',
                                       trainable=True)
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        input_dim = K.shape(x)[-1]
        v = K.tanh(K.dot(K.reshape(x, [-1, input_dim]), self.w_omega) + K.expand_dims(self.b_omega, 0))
        vu = K.dot(v, K.expand_dims(self.u_omega, -1))
        vu = K.reshape(vu, K.shape(x)[:2])
        m = K.cast(mask, dtype='float32')
        m = m - 1
        m = m * REMOVE_FACTOR
        vu = vu + m
        alphas = K.softmax(vu)
        output = K.sum(x * K.expand_dims(alphas, -1), 1)
        if self.return_alphas:
            return [output] + [alphas]
        else:
            return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[2])
        if self.return_alphas:
            alphas_shape = [(input_shape[0], input_shape[1])]
            return [output_shape] + alphas_shape
        else:
            return output_shape

    def get_config(self):
        return super(Attention, self).get_config()


class coAttention_para(Layer):
    """
    self-defined parallel co-attention layer.
    inputs: [tFeature, iFeature]
    outputs: [coFeature]

    dimension:
    input dimensions: [(batch_size, seq_length, embedding_size), (batch_size, num_img_region, 2*hidden_size)]
        considering subsequent operation, better to set embedding_size == 2*hidden_size
    output dimensions:[(batch_size, 2*hidden_size)]
    """
    def __init__(self, dim_k, **kwargs):
        super(coAttention_para, self).__init__(**kwargs)
        self.dim_k = dim_k  # internal tensor dimension
        self.supports_masking = True

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A Co-Attention_para layer should be called '
                             'on a list of inputs.')
        if len(input_shape) != 2:
            raise ValueError('A Co-Attention_para layer should be called on a list of 2 inputs.'
                             'Got '+str(len(input_shape))+'inputs.')
        self.embedding_size = input_shape[0][-1]
        self.num_region = input_shape[1][1]
        self.seq_len = input_shape[0][1]
        """
        naming variables following the VQA paper
        """
        self.Wb = self.add_weight(name="Wb",
                                  initializer="random_normal",
                                  # initializer="ones",
                                  shape=(self.embedding_size, self.embedding_size),
                                  trainable=True)
        self.Wq = self.add_weight(name="Wq",
                                  initializer="random_normal",
                                  # initializer="ones",
                                  shape=(self.embedding_size, self.dim_k),
                                  trainable=True)
        self.Wv = self.add_weight(name="Wv",
                                  initializer="random_normal",
                                  # initializer="ones",
                                  shape=(self.embedding_size, self.dim_k),
                                  trainable=True)
        self.Whv = self.add_weight(name="Whv",
                                   initializer="random_normal",
                                   # initializer="ones",
                                   shape=(self.dim_k, 1),
                                   trainable=True)
        self.Whq = self.add_weight(name="Whq",
                                   initializer="random_normal",
                                   # initializer="ones",
                                   shape=(self.dim_k, 1),
                                   trainable=True)

        super(coAttention_para, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        tFeature = inputs[0]
        iFeature = inputs[1]
        # affinity matrix C
        affi_mat = K.dot(tFeature, self.Wb)
        affi_mat = K.batch_dot(affi_mat, K.permute_dimensions(iFeature, (0, 2, 1)))  # (batch_size, seq_len, num_region)
        # Hq, Hv, av, aq
        tmp_Hv = K.dot(tFeature, self.Wq)
        Hv = K.dot(iFeature, self.Wv) + K.batch_dot(K.permute_dimensions(affi_mat, (0, 2, 1)), tmp_Hv)
        Hv = K.tanh(Hv)
        av = K.softmax(K.squeeze(K.dot(Hv, self.Whv), axis=-1))

        tmp_Hq = K.dot(iFeature, self.Wv)
        Hq = K.dot(tFeature, self.Wq) + K.batch_dot(affi_mat, tmp_Hq)
        Hq = K.tanh(Hq)
        aq = K.softmax(K.squeeze(K.dot(Hq, self.Whq), axis=-1))

        av = K.permute_dimensions(K.repeat(av, self.embedding_size), (0, 2, 1))
        aq = K.permute_dimensions(K.repeat(aq, self.embedding_size), (0, 2, 1))

        tfeature = K.sum(aq * tFeature, axis=1)
        ifeature = K.sum(av * iFeature, axis=1)

        return tfeature+ifeature

    def get_config(self):
        return super(coAttention_para, self).get_config()

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][-1])
        return output_shape


def myLossFunc(y_true, y_pred):
    probs_log = -K.log(y_pred)
    loss = K.mean(K.sum(probs_log*y_true, axis=-1))
    return loss


def tagOffSet(tags, index_from):
    tags = [x+index_from for x in tags]
    return [1] + tags


def zero_padding(X, seq_length):
    X_ = []
    for x in X:
        row = list(x)[:seq_length] + [0] * max(seq_length-len(x), 0)
        X_.append(np.array(row)*1.0)
    return np.array(X_).astype(int)


