# Speech emotion recognition on MSP-IMPROV dataset
# feature: egemaps
# Algorithm: as described in Bagus TA et al., "RNN-based dimenisional SER",
#                            ASJ Autumn 2019.
# 2012-12-24: Modified from ser_improv_egamaps, change input to HFS
#             Filter only natural interaction and improvisation data
# 2012-12-25: Updaate to adapt train/test data-label, change it accordingly
#             to be run in csl directory (although data in improv_loso/)

import numpy as np
import pandas as pd

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, TimeDistributed, Bidirectional, Flatten, \
                         Embedding, Dropout, Flatten, CuDNNLSTM, BatchNormalization, \
                         RNN, SimpleRNN, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from keras.preprocessing import sequence

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# load train and test data
feat_path = '/home/s1820002/msp-improv/egemaps_improv/'
list_path_train = '/home/s1820002/csl/improv_loso/data/data_mspin_train.csv'
list_file = pd.read_csv(list_path_train, index_col=None)

list_path_test = '/home/s1820002/csl/improv_loso/data/data_mspin_test.csv'
list_test = pd.read_csv(list_path_test, index_col=None)

vad = np.load('improv_loso/data/vad.npy')
list_sorted = pd.concat([list_file, list_test])

feat = []
for fn in list_sorted['id']:
    print("Process..", fn)
    data = pd.read_csv(feat_path+fn+'.csv', sep=';', usecols=range(2,25))
    feat.append(data)

feat = sequence.pad_sequences(feat)

print('Feature shape: ', feat.shape)
print('Label shape: ', vad.shape)

## standardization
scaled_vad = True

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


# dropout factor
do = 0.3


# API model, if use RNN, first two rnn layer must return_sequences=True
def api_model():
    inputs = Input(shape=feat[0].shape, name='feat_input')
    net = BatchNormalization()(inputs)
    net = CuDNNLSTM(256, return_sequences=True)(net)
    net = CuDNNLSTM(256, return_sequences=True)(net)
    net = CuDNNLSTM(256, return_sequences=True)(net)
    net = Flatten()(net)
    net = Dropout(do)(net)
    target_names = ('v', 'a', 'd')
    outputs = [Dense(1, name=name)(net) for name in target_names]

    model = Model(inputs=inputs, outputs=outputs) #=[out1, out2, out3])
    model.compile(loss=ccc_loss, #{'v': ccc_loss, 'a': ccc_loss, 'd': ccc_loss},
                  loss_weights={'v': 0.3, 'a': 0.6, 'd': 0.1},
                  optimizer='rmsprop', metrics=[ccc])
    return model


model2 = api_model()
model2.summary()

split = 5744      # train, train+dev = 5744

earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
hist = model2.fit(feat[:split], vad[:split].T.tolist(), batch_size=8,
                  validation_split=0.2, epochs=50, verbose=1, shuffle=True,
                  callbacks=[earlystop])

# outs =loss, mse_v, mse_a, mse_d, ccc_v, ccc_a, ccc_d
# test data start from 5744
metrik2 = model2.evaluate(feat[split:], vad[split:].T.tolist())
print(metrik2)

## make prediction
#feat_predict = feat[4596:]
#vad_predict = vad[4596:]

predict2 = model2.predict(feat[4596:], batch_size=8)
np.save('result_mspin_loso/result_ser.npy',  np.array(predict2).reshape(3, 2570).T)
