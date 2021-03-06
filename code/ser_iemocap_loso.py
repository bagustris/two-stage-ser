# CSL Paper: Dimensional speech emotion recognition from acoustic and text
# Changelog:
# 2019-09-01: initial version
# 2019-10-06: optimizer MTL parameters with linear search (in progress)
# 2019-12-25: modified fot ser_iemocap_loso_hfs.py
#             feature is either std+mean or std+mean+silence (uncomment line 44)
# 2020-05-26: remove outlier

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

from keras.callbacks import EarlyStopping, ModelCheckpoint

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# load feature and labels
# feat = np.load('/home/s1820002/spro2020/data/feat_ws_3.npy')
feat = np.load('/home/s1820002/IEMOCAP-Emotion-Detection/X_egemaps.npy')
vad = np.load('/home/s1820002/IEMOCAP-Emotion-Detection/y_egemaps.npy')

# remove outlier
outlier = np.array([1674, 3427, 5086, 5093, 5096, 5104, 7821])
mask = np.ones(len(vad), np.bool)
mask[outlier] = 0
vad = vad[mask]
feat = feat[mask]

# standardization, for LLD only
scaled_feature = False

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaler.transform(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaled_feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2])
    feat = scaled_feat
else:
    feat = feat

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
def api_model(alpha, beta, gamma):
    # speech network
    input_speech = Input(shape=(feat.shape[1], feat.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    net_speech = CuDNNLSTM(256, return_sequences=True)(net_speech)
    net_speech = CuDNNLSTM(256, return_sequences=True)(net_speech)
    net_speech = CuDNNLSTM(256, return_sequences=True)(net_speech)
    net_speech = Flatten()(net_speech)
    net_speech = Dropout(0.3)(net_speech)

    target_names = ('v', 'a', 'd')
    model_combined = [Dense(1, name=name)(net_speech) for name in target_names]

    model = Model(input_speech, model_combined) 
    model.compile(loss=ccc_loss, 
                  loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  optimizer='rmsprop', metrics=[ccc])
    return model


# def main(alpha, beta, gamma):
model = api_model(0.1, 0.5, 0.4)
model.summary()

# 7862 first data is 5 sessions (for LOSO)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                          restore_best_weights=True)
hist = model.fit(feat[:7862], vad[:7862].T.tolist(), batch_size=8, #best:8
                  validation_split=0.2, epochs=50, verbose=1, shuffle=True,
                  callbacks=[earlystop])
# don't use batch size here
metrik = model.evaluate(feat[7862:], vad[7862:].T.tolist())
print(metrik)

# save prediction, comment to avoid overwriting
# 6290: start of dev, 3742: total-dev
predict = model.predict(feat[6290:], batch_size=8)
np.save('result_iemocap_loso/result_ser', 
         np.array(predict).reshape(3, 3742).T)
