import os
import numpy as np
import pickle as pcl

import sys
sys.path.insert(0, '..')
from predict import predict


si = predict(fname='../data/full_histories_illustris.h5')
si.training_mask()
si.filename

si.generate_standardisation('Intrinsic')
features = si.prepare_features(key='Intrinsic', CNN=True)
predictors = si.load_arr('log_8','SFH')

model = si.create_cnn_model(features, predictors, batch_size=20, train=si.train, fit=False)

from keras.utils import plot_model
plot_model(model, to_file='model.png')

# prediction = model.predict(features[~si.train])
# test_score = np.mean([si._SMAPE(predictors[~si.train][i], prediction[i]) \
#                              * 100 for i in range(len(prediction))])
# 
# print("Test score: %d"%test_score)
# 
# f = '../data/cnn_trained_illustris_intrinsic.h5'
# if os.path.isfile(f): os.remove(f)
# model.save(f)

