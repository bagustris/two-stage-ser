# CSL Paper: Dimensional speech emotion recognition from acoustic and text
# Changelog:
# 2019-09-01: initial version
# 2019-10-06: optimizer MTL parameters with linear search (in progress)
# 2019-12-21: update file for CSL journal
#             this python use following feature: mean, std, silence

import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, CuDNNLSTM, Embedding, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# load data - label
list_path_train = '/home/s1820002/csl/improv_loso/data/data_mspin_train.csv'
list_train = pd.read_csv(list_path_train, index_col=None)

vad_train = [list_train['v'], list_train['a'], list_train['d']]
vad_train = np.array(vad_train).T

list_path_test = '/home/s1820002/csl/improv_loso/data/data_mspin_test.csv'
list_test = pd.read_csv(list_path_test, index_col=None)

vad_test = [list_test ['v'], list_test['a'], list_test['d']]
vad_test = np.array(vad_test).T

vad = np.concatenate([vad_train, vad_test])

# load Glove pretrained model
g_word_embedding_matrix = np.load('improv_loso/data/g_word_embedding_matrix.npy', allow_pickle=True)

EMBEDDING_DIM = 300
nb_words = 3428
MAX_SEQUENCE_LENGTH = 300

# print input size
print('Feature shape: ', x_train_text.shape)
print('Label shape: ', vad_train.shape)

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


# Concordance correlation coefficient (CCC)-based loss function
# using non-inductive statistics
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


# split train/test
split = 5375
earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
                          restore_best_weights=True)

# API model, if use RNN, first two rnn layer must return_sequences=True
# model: GRU
def text_model1():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights=[g_word_embedding_matrix],
                    trainable=True)(inputs)
    net = CuDNNLSTM(256, return_sequences=True)(net)
    net = CuDNNLSTM(256, return_sequences=True)(net)
    net = CuDNNLSTM(256, return_sequences=False)(net)
    net = Dense(64)(net)
    net = Dropout(0.3)(net)

    target_names = ('v', 'a', 'd')
    outputs = [Dense(1, name=name)(net) for name in target_names]

    model = Model(inputs=inputs, outputs=outputs) #=[out1, out2, out3])
    model.compile(loss=ccc_loss, #{'v': ccc_loss, 'a': ccc_loss, 'd': ccc_loss},
                  loss_weights={'v': .1, 'a': .6, 'd': .3},
                  optimizer='rmsprop', metrics=[ccc])
    return model


model1 = text_model1()
hist1 = model1.fit(x_train_text, vad_train.T.tolist(), epochs=50, 
                   batch_size=32, verbose=1, validation_split=0.2, 
                   callbacks=[earlystop])

eval_metrik1 = model1.evaluate(x_test_text, vad_test.T.tolist())
print(eval_metrik1)

# make prediction
predict = model1.predict(x_text[4596:], batch_size=32)
np.save('result_mspin_loso/result_ter_glove', 
         np.array(predict).reshape(3, 2570).T)