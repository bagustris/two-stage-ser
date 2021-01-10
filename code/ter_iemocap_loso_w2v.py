# OCOCOSDA 2019: dimensional speech emotion recognition from text feature

# uncomment these to run on CPU only
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras.backend as K

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Embedding, Dropout, Flatten, concatenate, CuDNNGRU
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping

from gensim.models.keyedvectors import KeyedVectors

np.random.seed(99)

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


# text feature and labels
path = '/home/s1820002/IEMOCAP-Emotion-Detection/'
x_train_text = np.load(path+'x_train_text.npy')
vad = np.load(path+'y_egemaps.npy')

# remove outlier
outlier = np.array([1674, 3427, 5086, 5093, 5096, 5104, 7821])
mask = np.ones(len(vad), np.bool)
mask[outlier] = 0
vad = vad[mask]
x_train_text = x_train_text[mask]

#other parameters, determined from previous experiment
MAX_SEQUENCE_LENGTH = 554
EMBEDDING_DIM = 300
nb_words = 3438

file_loc = path + 'GoogleNews-vectors-negative300.bin'
print (file_loc)

word_vectors = KeyedVectors.load_word2vec_format(file_loc, binary=True)

g_word_embedding_matrix = {}
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i>=nb_words:
        continue
    try:
        gembedding_vector = word_vectors[word]
        g_word_embedding_matrix[i] = gembedding_vector
    except KeyError:
        g_word_embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

del(word_vectors)

scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    vad = scaled_vad 
else:
    vad = vad

# split train/test
split = 7869
earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
                          restore_best_weights=True)

# model: GRU
def text_model1():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
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
                  loss_weights={'v': 0.7, 'a': 0.2, 'd': 0.1},
                  optimizer='rmsprop', metrics=[ccc])
    return model

model1 = text_model1()
hist1 = model1.fit(x_train_text[:split], vad[:split].T.tolist(), epochs=50, 
                   batch_size=8, verbose=1, validation_split=0.2, 
                   callbacks=[earlystop])

eval_metrik1 = model1.evaluate(x_train_text[split:], vad[split:].T.tolist())
print(eval_metrik1)

# make prediction
# predict = model1.predict(x_train_text[6290:], batch_size=8)
# np.save('result_iemocap_loso/result_ter_w2v', 
#          np.array(predict).reshape(3, 3742).T)
