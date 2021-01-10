# CSL Paper: Dimensional speech emotion recognition from acoustic and text
# Changelog:
# 2019-09-01: initial version
# 2019-10-06: optimizer MTL parameters with linear search (in progress)
# 2019-12-21: update file for CSL journal
#             this python use following feature: mean, std, silence

import numpy as np
import pickle
import pandas as pd

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, \
                         Bidirectional, Flatten, \
                         Embedding, Dropout, Flatten, BatchNormalization, \
                         RNN, concatenate, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

#from keras_self_attention import SeqSelfAttention
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from attention_helper import AttentionDecoder
#from read_csv import load_features

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# load label
list_path = '/home/s1820002/msp-improv/data/data_mspin.csv'
list_file = pd.read_csv(list_path, index_col=None)
list_sorted = list_file.sort_values(by=['id'])

vad_list = [list_sorted['v'], list_sorted['a'], list_sorted['d']]
vad = np.array(vad_list).T
#vad = np.nan_to_num(vad, nan=3)

## Text processing
text = list_sorted['transcript']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
MAX_SEQUENCE_LENGTH = len(max(token_tr_X, key=len))
x_train_text = []

x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

EMBEDDING_DIM = 300

word_index = tokenizer.word_index
nb_words = len(word_index) + 1
print('Found %s unique tokens' % len(word_index))

# print input size
print('Feature shape: ', x_train_text.shape)
print('Label shape: ', vad.shape)

# standardization
scaled_feature = False

# set Dropout
do = 0.3

# standardized label
scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    vad = scaled_vad 
else:
    vad = vad

# Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
def ccc(gold, pred):
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    return ccc


def ccc_loss(gold, pred):  
    # input (num_batches, seq_len, 1)
    ccc_loss   = K.constant(1.) - ccc(gold, pred)
    return ccc_loss


# API model, if use RNN, first two rnn layer must return_sequences=True
# split train/test
split = 5375
earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
                          restore_best_weights=True)

# model: GRU
def text_model1():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    trainable = True)(inputs)
    net = CuDNNLSTM(256, return_sequences=True)(net)
    net = CuDNNLSTM(256, return_sequences=True)(net)
    net = CuDNNLSTM(256, return_sequences=False)(net)
    net = Dense(128)(net)
    net = Dropout(0.3)(net)

    target_names = ('v', 'a', 'd')
    outputs = [Dense(1, name=name)(net) for name in target_names]

    model = Model(inputs=inputs, outputs=outputs) #=[out1, out2, out3])
    model.compile(loss=ccc_loss, #{'v': ccc_loss, 'a': ccc_loss, 'd': ccc_loss},
                  loss_weights={'v': 1, 'a': 1, 'd': 1},
                  optimizer='rmsprop', metrics=[ccc])
    return model


model1 = text_model1()
hist1 = model1.fit(x_train_text[:split], vad[:split].T.tolist(), epochs=30, 
                   batch_size=8, verbose=1, validation_split=0.2, 
                   callbacks=[earlystop])

eval_metrik1 = model1.evaluate(x_train_text[split:], vad[split:].T.tolist())
print(eval_metrik1)

# make prediction
predict = model1.predict(x_train_text[4586:], batch_size=32)
np.save('result_mspin_sd/result_ter_we', 
         np.array(predict).reshape(3, 2580).T)
