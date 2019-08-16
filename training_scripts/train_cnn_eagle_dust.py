import os
import numpy as np
import pickle as pcl

# import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=False)

import sys
# from pathlib import Path
# pth = Path(os.getcwd()).parent
# print(pth)
sys.path.insert(0, '..') # pth)
# import predict
from predict import predict


si = predict(fname='../data/full_histories_eagle.h5')
si.training_mask()
si.filename

si.generate_standardisation('Dust')
features = si.prepare_features(key='Dust', CNN=True)
predictors = si.load_arr('log_8','SFH')

model,scores = si.create_cnn_model(features, predictors, batch_size=20, train=si.train)

prediction = model.predict(features[~si.train])
test_score = np.mean([si._SMAPE(predictors[~si.train][i], prediction[i]) \
                             * 100 for i in range(len(prediction))])

print("Test score: %d"%test_score)

f = '../data/cnn_trained_eagle_dust.h5'
if os.path.isfile(f): os.remove(f)
model.save(f)

