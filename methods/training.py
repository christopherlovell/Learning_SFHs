import numpy as np
import pickle as pcl
import random

from keras.optimizers import Adam
from keras.models import Sequential

from keras.layers import Dense, InputLayer, Dropout, Activation
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Flatten, GlobalMaxPooling1D

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras.models import load_model

# from hyperas import optim
# from hyperas.distributions import choice, uniform, conditional

from keras.constraints import nonneg

# from hyperopt import Trials, STATUS_OK, tpe

from keras import backend as K

from methods.loss import _SMAPE, _SMAPE_tf, _R_squared_tf


def load_data(CNN=False):
    """
    Load data for training
    """
    
    pickle = pcl.load(open('data/full_histories_binned.p', 'rb'))

    wl = pickle['header']['Wavelength']
    wl_mask = (wl > 912) & (wl < 1e5)
    
    # features = np.log(np.array([value['SED'][wl_mask] for key, value in pickle['data'].items()]))
    features = np.log(np.array([value['SED_norm'] for key, value in pickle['data'].items()]))

    predictors = np.array([value['log']['SFH'] for key, value in pickle['data'].items()])

    train = pickle['header']['train']
    
    if CNN:
        if len(predictors.shape) == 1:
            predictors = np.reshape(predictors, (predictors.shape[0],1))  

        if len(features.shape) < 3:
            features.shape += (1,)

    return features, predictors, train, wl, wl_mask, pickle


# def create_cnn_model(features, predictors, train):
# 
#     input_dim = features.shape[1:]
#     out_dim = predictors.shape[1]
# 
#     filters = [13,20]
#     kernel_size = 8
#     # hidden_dims = [128,64,32,16]
#     hidden_dims = [512,256,128]
#     initializer = 'he_normal'
# 
#     lr = 0.0007
#     beta_1 = 0.9
#     beta_2 = 0.999
#     optimizer_epsilon = 1e-08
# 
#     optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)
# 
#     max_epochs = 1000
#     batch_size = 20
# 
#     early_stopping_min_delta = 0.00001
#     early_stopping_patience =  6
#     reduce_lr_factor = 0.5
#     reuce_lr_epsilon = 0.0009
#     reduce_lr_patience = 2
#     reduce_lr_min = 0.00008
# 
#     model = Sequential()
# 
#     model.add(Conv1D(filters=filters[0], 
#                      kernel_size=kernel_size,
#                      activation='relu',
#                      padding='same',
#                      input_shape=input_dim))
# 
#     model.add(Conv1D(filters=filters[1], 
#                      kernel_size=kernel_size,
#                      padding='same',
#                      activation='relu', strides=1))
# 
#     model.add(GlobalMaxPooling1D())
#     model.add(Dense(hidden_dims[0], kernel_initializer=initializer))
#     model.add(Dense(hidden_dims[1], kernel_initializer=initializer))
#     model.add(Dense(hidden_dims[2], kernel_initializer=initializer))
#     #model.add(Dense(hidden_dims[3], kernel_initializer=initializer))
#     model.add(Activation('relu'))
# 
#     model.add(Dense(out_dim, kernel_initializer='normal', kernel_constraint=nonneg()))
# 
#     model.compile(loss=_SMAPE_tf, optimizer=optimizer, metrics=['mae','mse','accuracy'])
#     # model.compile(loss=_R_squared_tf, optimizer=optimizer, metrics=['mae','mse','accuracy'])
# 
#     early_stopping = EarlyStopping(monitor='loss', min_delta=early_stopping_min_delta, 
#                                            patience=early_stopping_patience, verbose=2, mode='min')
# 
#     reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=reuce_lr_epsilon, 
#                                       patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)
# 
#     tfcallback = TensorBoard(log_dir='graphs/.', histogram_freq=0,  
#                    write_graph=True, write_images=True)
# 
#     history = model.fit(features[train], predictors[train],
#               callbacks=[early_stopping, reduce_lr, tfcallback], 
#               epochs=max_epochs, batch_size=batch_size, verbose=True)
# 
#     score, mae, mse, acc = model.evaluate(features[~train], predictors[~train], verbose=0)
#     print('Test SMAPE:', score)
#     return model, {'loss': score, 'mse': mse, 'mae': mae, 'acc': acc}#, 'history': history}
# 
