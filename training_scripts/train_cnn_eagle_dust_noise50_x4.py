import os
import numpy as np
import pickle as pcl

import sys
sys.path.insert(0, '..') # pth)
from predict import predict

print(os.path.basename(__file__))


si = predict(fname='../data/full_histories_eagle.h5')
si.training_mask()
si.filename

sn = 50
eagle_dust, wl = si.load_spectra('Dust')
eagle_dust_noise = si.add_noise_flat(spec=eagle_dust, wl=wl, sn=sn)
si.generate_standardisation(key='Dust Noise SN50', spec=eagle_dust_noise)
features = si.prepare_features(eagle_dust_noise, key='Dust Noise SN50', CNN=True)

predictors = si.load_arr('log_8','SFH')

# reasmple
count = 0
train = si.train
while count < 3:    
    eagle_dust_noise = si.add_noise_flat(spec=eagle_dust, wl=wl, sn=sn)
        
    feat = si.prepare_features(eagle_dust_noise, key="Dust Noise SN50", CNN=True)
    features = np.vstack([features, feat])
    
    predictors = np.vstack([predictors, predictors[:len(feat)]])
    train = np.hstack([train, train[:len(feat)]])
    
    count += 1
    print(count)


model,scores = si.create_cnn_model(features, predictors, batch_size=20, train=train)

prediction = model.predict(features[~train])
test_score = np.mean([si._SMAPE(predictors[~train][i], prediction[i]) \
                             * 100 for i in range(len(prediction))])

print("Test score: %d"%test_score)

f = '../data/cnn_trained_eagle_dust_noise50_x4.h5'
if os.path.isfile(f): os.remove(f)
model.save(f)

