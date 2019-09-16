import os
import numpy as np
import pickle as pcl

import sys
sys.path.insert(0, '..')
from predict import predict

si = predict(fname='../data/full_histories_eagle.h5')
si.training_mask()
si.filename

sn=50
illustris_intr, wl = si.load_spectra('Intrinsic')
illustris_intr_noise = si.add_noise_flat(spec=illustris_intr, wl=wl, sn=sn)
si.generate_standardisation(key='Intrinsic Noise SN50', spec=illustris_intr_noise)
features = si.prepare_features(illustris_intr_noise, key='Intrinsic Noise SN50', CNN=True)

predictors = si.load_arr('log_8','SFH')

# resampled
count = 0
train = si.train
while count < 3:    
    illustris_intr_noise = si.add_noise_flat(spec=illustris_intr, wl=wl, sn=sn)
        
    feat = si.prepare_features(illustris_intr_noise, key="Intrinsic Noise SN50", CNN=True)
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

f = '../data/cnn_trained_eagle_intrinsic_noise50_x4.h5'
if os.path.isfile(f): os.remove(f)
model.save(f)

