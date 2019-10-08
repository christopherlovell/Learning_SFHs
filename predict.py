import numpy as np
import h5py

import astropy.units as u
from astropy.cosmology import z_at_value

from spectacle.spectacle import spectacle

from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline

## keras 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, InputLayer, Dropout, Activation
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Flatten, GlobalMaxPooling1D, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from tensorflow.keras.constraints import NonNeg

class predict(spectacle):

    def __init__(self, **kwargs):
        """
        Predict SFH for galaxies within a `sph2sed.model.sed` class

        Args:
            sed (instance) sed class instance
        """
        spectacle.__init__(self, **kwargs)
        self.scalers = {}
        self.training_mask()



    def training_mask(self, frac=0.2):
        
        shids = self.load_arr('ID','Subhalos')

        if shids is not None:

            with h5py.File(self.filename,'a') as f:
                if 'Train' in f.keys():
                    if len(f['Train'][:]) != len(shids):
                        raise ValueError('Error: Training mask and Subhalo arrays are not same size')
                    else:
                        self.train = f['Train'][:]
                else:
                    train = np.random.rand(len(shids)) > frac
                    f.create_dataset('Train',(len(train),),maxshape=(None,),data=train)
                    self.train = train
        else:
            print("No Subhalos, failed to make training mask")
            return 0

        
    def generate_standardisation(self, key, spec=None, train=True):
        """
        Args:
         key (str) spectra key
         train (bool) use just the training data (defined by the train mask) to generate scaler, or whole data set
        """

        if spec is None:
            spec, wl = self.load_spectra(key)

        if train:
            if self.train is None:
                raise ValueError('No training mask generated')
            else:
                spec = spec[self.train]

        f_trans = FunctionTransformer(np.log10, validate=True)
        pipe = make_pipeline(f_trans, StandardScaler())

        if "scalers" not in self.__dict__:
            self.scalers = {}

        self.scalers[key] = pipe.fit(spec) 



    def apply_standardisation(self, key):
        spec, wl = self.load_spectra(key)
        return self.scaler[key].transform(spec)  



    def init_bins(self, binning='linear', name='linear', Nbins=20, Nlow=7.3, custom=None, verbose=False):
        
        if custom is not None:
            if isinstance(custom, dict):
                bins = custom['bins']
                binLimits = custom['binLimits']
                binWidths = custom['binWidths']
            else:
                raise ValueError("Must specify dict with 'bins', 'binLimits' and 'binWidths' parameters")
        else:  # initialise bins 
            if binning == 'linear':
                binLimits = np.linspace(0, self.cosmo.age(self.redshift).value, Nbins+1)
                binWidth = binLimits[1] - binLimits[0]
                bins = np.linspace(binWidth/2, binLimits[-1] - binWidth/2, Nbins)
                binWidths = binWidth * 1e9        
            elif binning == 'log':
                upperBin = np.log10(self.cosmo.age(self.redshift).value * 1e9)
                binLimits = np.hstack([[0.0], np.linspace(Nlow, upperBin, Nbins)])
                binWidths = 10**binLimits[1:] - 10**binLimits[:len(binLimits)-1]
                bins = binLimits[:-1] + ((binLimits[1:] - binLimits[:-1]) / 2)
                binLimits = 10**binLimits / 1e9
            else:
                raise ValueError('Invalid binning chosen, use either \'linear\' or \'log\'')
            
        if verbose: print("Saving bin info to file...")
        self.save_arr(arr=bins,name='bins',group='bins/%s'%name)
        self.save_arr(arr=binLimits,name='binLimits',group='bins/%s'%name)
        self.save_arr(arr=binWidths,name='binWidths',group='bins/%s'%name)

        return bins, binLimits, binWidths


    def bin_histories(self, shid, binLimits, binWidths):
        """
        Bin star formation histories from particle age data

        Args:
            binning (str) linear or log
            name (str) linear or log
            Nbins (int) number of bins
            Nlow (float) if log binning chosen, the log age of the second lower bin limit
        """

        ftime,met,imass = self.load_galaxy(shid)
            
        # Add redshift to give age of stars from point of observation
        binLimits_temp = binLimits + self.cosmo.lookback_time(self.redshift).value

        # first set age limits so that z calculation doesn't fail
        binLimits_temp[binLimits_temp < 1e-6] = 1e-6
        binLimits_temp[binLimits_temp > (self.cosmo.age(0).value - 1e-3)] = (self.cosmo.age(0).value - 1e-3)

        # convert binLimits to scale factor
        binLimits_temp = self.cosmo.scale_factor([z_at_value(self.cosmo.lookback_time, a) for a in binLimits_temp * u.Gyr])

        # weights converts age to SFR (bins must increase monotonically, so reverse for now)
        counts, dummy = np.histogram(ftime, bins=binLimits_temp[::-1], weights=imass);

        ## divide by bin width in (Giga)years to give SFR in Msol / year
        # (reverse counts to fit original)
        return counts[::-1] / binWidths



#     def prepare_predictors(self, binning='log_6', CNN=False):
#         """
#         Prepare star formation histories for input in to model

#         Args:
#             binning (str) log or linear
#             CNN (bool) whether to transform shape suitable for input to keras CNN
#         """
        
#         predictors = np.array([value['SFH'][binning] for key, value in self.galaxies.items()])

#         # if CNN:
#         #     if len(predictors.shape) < 3:
#         #         predictors = np.reshape(predictors, predictors.shape + (1,))

#         return predictors


    def prepare_features(self, spec=None, key='Intrinsic', scaler=None, CNN=False, verbose=False):
        """
        Prepare spectra for input to predictive model

        Args:
            spectra (str) spectra key
            scaler (object) scaler object to apply to features. If not specified, default from self.spectra[spectra]['scaler'] used
            CNN (bool) whether to transform shape suitable for input to keras CNN
            verbose (bool) verbosity flag
        """

        if spec is None:
            spec, wl = self.load_spectra(key)
#         features = np.array([value['Spectra'][spectra] for key, value in self.galaxies.items()])

        if scaler is None:
            if verbose: print("Using default scaler")
            
            if key not in self.scalers.keys():
                self.scalers[key] = None

            if self.scalers[key] is None:
                self.generate_standardisation(key=key,spec=spec)
                #raise ValueError('Default scaler not defined')
            scaler = self.scalers[key]
        
        features = scaler.transform(spec)

        if CNN:
            if len(features.shape) < 3:
                features.shape += (1,)
        
        return features



    def create_cnn_model(self, features, predictors, batch_size=10, train=None, plot=False, max_epochs=1000, loss=None, verbose=False, fit=True):
        """
        Define, initialise and train CNN

        Args:
            features (array) 
            predictors (array) 
        """

        if train is None:
            if self.train is None: raise ValueError('No training mask initialised')
            train = self.train

        if loss is None:
            loss = self._SMAPE_tf
       
        # updatable plot
        if plot:
            ## live plotting
            import matplotlib
            matplotlib.use('agg')
            from matplotlib import pyplot as plt
            from IPython.display import clear_output
            class PlotLosses(Callback):
                def on_train_begin(self, logs={}):
                    self.i = 0
                    self.x = []
                    self.losses = []
                    self.val_losses = []
                
                    self.fig = plt.figure()
                
                    self.logs = []
        
                def on_epoch_end(self, epoch, logs={}):
                
                    self.logs.append(logs)
                    self.x.append(self.i)
                    self.losses.append(logs.get('loss'))
                    self.val_losses.append(logs.get('val_loss'))
                    self.i += 1
                

                    clear_output(wait=True)
                    plt.plot(self.x, self.losses, label="loss")
                    plt.plot(self.x, self.val_losses, label="val_loss")
                    plt.legend()
                    plt.show();
                

        input_dim = features.shape[1:]
        out_dim = predictors.shape[1]
    
        initializer = 'he_normal'
    
        model = Sequential()
    
        model.add(Conv1D(filters=35, 
                         kernel_size=25,
                         padding='same',
                         input_shape=input_dim))

        model.add(Activation(activation='relu'))
        model.add(BatchNormalization())
    
        model.add(Conv1D(filters=55, 
                         kernel_size=55,
                         padding='same'))
        
        model.add(Activation(activation='relu'))
        model.add(BatchNormalization())
    
        model.add(GlobalMaxPooling1D())
        
        model.add(Dense(55, kernel_initializer=initializer))
        model.add(Dense(55, kernel_initializer=initializer))

        model.add(Activation('relu'))
    
        model.add(Dense(out_dim, 
                        kernel_initializer='normal', 
                        kernel_constraint=nonneg()))
    
        lr = 0.0007
        beta_1 = 0.9
        beta_2 = 0.999
    
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=0.0)
    
        model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=['mae','mse','accuracy'])
    
        early_stopping_min_delta = 1e-4
        early_stopping_patience = 12
        
        early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=early_stopping_min_delta,
                                   patience=early_stopping_patience,
                                   verbose=2, mode='min')

        reduce_lr_patience = 8
        reduce_lr_min = 0.0
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=reduce_lr_patience,
                                  min_lr=reduce_lr_min,
                                  mode='min', verbose=2)
  
        tfcallback = TensorBoard(log_dir='graphs/.', histogram_freq=0,  
                       write_graph=True, write_images=True)
    
        if plot:
            plot_losses = PlotLosses() 
            callbacks = [early_stopping, reduce_lr, tfcallback, plot_losses]
        else:
            callbacks = [early_stopping, reduce_lr, tfcallback]


        # validation split is unshuffled, so need to pre-shuffle before training
        mask = np.random.permutation(np.sum(train))

        if fit:
            history = model.fit(features[train][mask], predictors[train][mask],
                                callbacks=callbacks, epochs=max_epochs, 
                                batch_size=batch_size, validation_split=0.2, 
                                verbose=verbose)
    
            score, mae, mse, acc = model.evaluate(features[~train], predictors[~train], verbose=0)
            print('Test SMAPE:', score)
            return model, {'loss': score, 'mse': mse, 'mae': mae, 'acc': acc, 'history': history}
        else:
            return model


    ## TODO: check!! (particularly scalers)
    def _propogate_uncertainties(self, model, key, scaler, N=10, sn=50):
        """
        Propogate spectral uncertainties through model

        Args:
            model (keras model) CNN model
            key (str) 
            scaler (Keras scaler object) should be the scaller for the noise-added spectra case
            N (int) number of resamples
            sn (int) signal to noise ratio
    
        Returns:
            pred (array, [N,n,b]) where n is number of galaxies, and b number of bins
        """

        pred = np.array([None] * N) 
        
        spec, wl = self.load_spectra(key);

        for i in np.arange(N):
            spec_noise = self.add_noise_flat(spec, wl, sn=sn)
            spec_noise = self.prepare_features(spec_noise, scaler=scaler, CNN=True)

            pred[i] = model.predict(spec_noise[~self.train])


        pred = np.stack(pred)

        return pred


    def calculate_uncertainties(self, pred):
        """
        Use resampled histories to estimate sigma and the correlation matrix
        
        Args:
            pred (array, N x n x b) where N is number of samples, 
                                    n is number of galaxies, 
                                    and b is number of bins
        """
        cov = [np.cov(pred[:,i,:].T) for i in np.arange(pred.shape[1])]
        cov = np.stack(cov)

        sigma = [np.sqrt(np.diag(cov[i])) for i in np.arange(pred.shape[1])]
        sigma= np.stack(sigma)

        corr = cov / sigma[:,:,None] / sigma[:,None,:]
        
        return sigma, corr
    

    def evaluate_uncertainties(self,pred,true_sfh):
       
        percentile = np.array([None] * len(true_sfh), dtype=np.float32)
        for b,sfh in enumerate(true_sfh):

            hist, bin_limits = np.histogram(pred[:,b], bins=100,
                    range=(pred[:,b].min()-0.1,pred[:,b].max()+0.1))

            bins = bin_limits[:-1] - (bin_limits[1] - bin_limits[0])/2.
            cumulative = np.cumsum(hist) / pred.shape[0]
            percentile[b] = cumulative[np.argmin(np.abs(bins - sfh))]

        return percentile
 


    @staticmethod     
    def mse(true, pred):
        """
        Mean squared error
        """
        return np.sum(pow(true - pred, 2)) / len(true)
    
    @staticmethod     
    def mse_tf(true, pred):
        """
        Mean squared error
        """
        return K.sum((true - pred)**2, axis=-1) / K.shape(true)[1]

    @staticmethod     
    def mae(true, pred):
        """
        Mean absolute error
        """
        return np.sum(np.abs(true - pred)) / len(true)
    
    @staticmethod     
    def mae_tf(true, pred):
        """
        Mean absolute error
        """
        return K.sum(K.abs(true - pred), axis=-1) / len(true)

    @staticmethod     
    def absolute_error(true, pred):
        """
        Absolute error
        """
        return np.sum(np.abs(true - pred))
    
    @staticmethod     
    def absolute_error_tf(true, pred):
        """
        Absolute error
        """
        return K.sum(K.abs(true - pred), axis=-1)

    @staticmethod     
    def _SMAPE(y_true, y_pred):
        """
        Symmetric Mean Absolute Percentage Error
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        return 2 * np.sum(np.abs(y_pred - y_true), axis=-1) / np.sum(y_pred + y_true, axis=-1)

    @staticmethod     
    def _R_squared(y_true, y_pred):
        """
        R^2 error
        """
        return 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2)

    @staticmethod     
    def _SMAPE_tf(y_true, y_pred):
        """
        Symmetric Mean Absolute Percentage Error
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        return 2 * K.sum(K.abs(y_pred - y_true), axis=-1) / K.sum(y_pred + y_true, axis=-1)
    
    @staticmethod     
    def _SMAPE_sq_tf(y_true, y_pred):
        """
        Symmetric Mean Absolute Percentage Error
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        return 2 * K.sum(K.abs(y_pred - y_true)**2, axis=-1) / K.sum(y_pred + y_true, axis=-1)


    @staticmethod     
    def _SMAPE_weighted_tf(y_true, y_pred):
        """
        Symmetric Mean Absolute Percentage Error
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        return K.sum(K.abs(y_pred - y_true) * K.variable([1,.9,.8,.7,.6,.5,.5,.5,.5,.5]), axis=-1) / K.sum(y_pred + y_true, axis=-1)
    
    @staticmethod     
    def _SMAPE_plus_tf(y_true, y_pred):
        """
        Symmetric Mean Absolute Percentage Error
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        perc_error = (K.abs(K.sum(y_pred) - K.sum(y_true)) / K.sum(y_true))
        
        smape =  K.sum(K.abs(y_pred - y_true), axis=-1) / K.sum(y_pred + y_true, axis=-1) 
    
        return K.sqrt(K.square(perc_error) + K.square(smape))


    @staticmethod     
    def _R_squared_tf(y_true, y_pred):
        """
        negative R^2 error
        """
        return -1 * (1 - K.sum((y_pred - y_true)**2, axis=-1) / K.sum((y_true - K.mean(y_true))**2, axis=-1))
    
