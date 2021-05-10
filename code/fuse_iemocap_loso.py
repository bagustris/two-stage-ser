# simple_fuse: simple fusion from acoustic and txt
# run in MPC as : exec(open('simple_fuse_dev.py').read())
# 2019-12-20: initial work, it works!
# 2019-12-21: update to use silence feature from speech network
#             new data splitting (6000/2000/2039)

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from calc_scores import calc_scores

# option: ser, ser_hfs, ser_ws, ter, ter_w2v, ter_glove, is saved change the name
speech = 'ser_ws'   # ser, ser_hfs, ser_ws
text = 'ter_glove'        # ter, ter_w2v, ter_glove'
ser = np.load('result_iemocap_loso/result_' + speech + '.npy')
ter = np.load('result_iemocap_loso/result_' + text + '.npy')


# split dev and test
split = 1572
ser_dev = ser[:split]
ser_test = ser[split:]
ter_dev = ter[:split]
ter_test = ter[split:]

# load label 
vad = np.load('/home/s1820002/IEMOCAP-Emotion-Detection/y_egemaps.npy')

# remove outlier
outlier = np.array([1674, 3427, 5086, 5093, 5096, 5104, 7821])
mask = np.ones(len(vad), np.bool)
mask[outlier] = 0
vad = vad[mask]

scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad) 
    scaled_vad = scaler.transform(vad) 
    vad = scaled_vad 
else:
    vad = vad

y_dev = vad[6290:7862]
y_test = vad[7862:] 

# SVR model
svr_rbf = SVR(kernel='rbf', C=200, gamma=0.1, epsilon=0.01)

# predicting valence
valence_model = svr_rbf.fit(np.array([ter_dev[:,0], ser_dev[:,0]]).T, y_dev[:,0])
valence_pred = valence_model.predict(np.array([ter_test[:,0], ser_test[:,0]]).T)
ccc_v, pcc_v, rmse_v = calc_scores(valence_pred, y_test[:,0])

# predicting arousal
arousal_model = svr_rbf.fit(np.array([ter_dev[:,1], ser_dev[:,1]]).T, y_dev[:,1])
arousal_pred = valence_model.predict(np.array([ter_test[:,1], ser_test[:,1]]).T)
ccc_a, pcc_a, rmse_a = calc_scores(arousal_pred, y_test[:,1])

# predicting dominance
dominance_model = svr_rbf.fit(np.array([ter_dev[:,2], ser_dev[:,2]]).T, y_dev[:,2])
dominance_pred = valence_model.predict(np.array([ter_test[:,2], ser_test[:,2]]).T)
ccc_d, pcc_d, rmse_d = calc_scores(dominance_pred, y_test[:,2])

print('CCC', ccc_v, ccc_a, ccc_d)
print('PCC', pcc_v, pcc_a, pcc_d)
print('RMSE', rmse_v, rmse_a, rmse_d)

vad_pred = np.vstack((valence_pred, arousal_pred, dominance_pred)).T
filename = 'svm_iemocap_loso/' + speech + '_' + text + '.npy' 
np.save(filename, vad_pred)