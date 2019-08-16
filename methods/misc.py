import numpy as np
import copy
import random

from .astro import scale_factor_to_z

from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

import matplotlib.pyplot as plt


def total_error(sfh, experiment_err, model_err, key='intrinsic', fracs=False):
    """
    quadrature sum of model + spectra error
    """
    
    sigma = [None] * len(sfh)
    for i in np.arange(len(sfh)):
        m2,m1,c = experiment_err[key][i]
        sigma[i] = m2*np.log10(sfh[i])**2 + m1*np.log10(sfh[i]) + c
    
    sigma = np.array(sigma)
    # print(sigma)
    # print(sfh * model_err[key])
    
    total_error = np.sqrt(sigma**2 + (sfh * model_err[key])**2)
    
    if fracs:
        return total_error, sigma, sfh * model_err[key]
    else:
        return total_error


    
def add_y_eq_x(ax):
    """
    Add y = x line to axis

    Args:
    ax: axis object

    Returns:
    None
    """

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.5, linestyle='dashed');
    ax.set_aspect('equal')
    ax.set_xlim(lims);
    ax.set_ylim(lims);
    

def pickle_age_calc(pickle, log=False, cosmo=cosmo):
    """
    Calculate age and formation_time parameters for a given pickle of SFHs, with bin and binLimit header info
    """
    
    binning = 'log' if log else 'linear'
        
    binLimits = pickle['header'][binning]['binLimits']
    bins = pickle['header'][binning]['bins']

    pickle['header'][binning]['binWidth'] = binLimits[1] - binLimits[0]
    
    if log:
        pickle['header'][binning]['age'] = 10**bins / 1e9
    else:
        pickle['header'][binning]['age'] = bins
    
    pickle['header'][binning]['formation_time'] = cosmo.age(0).value - pickle['header'][binning]['age']
    
    return pickle


def generate_mocks(pickle, N):
    
    max_key = np.array(list(pickle['data'].keys())).max()
    
    original_keys = list(pickle['data'])
    
    for i in range(N):
        
        # create a new key
        key = max_key + i
    
        # initialise dict
        pickle['data'][key] = {}
        pickle['data'][key]['history'] = {}
        
        # choose two random original histories
        key_A = random.choice(original_keys)
        key_B = random.choice(original_keys)
        
        InitialStellarMass = np.hstack([pickle['data'][key_A]['history']['InitialStellarMass'], \
                                        pickle['data'][key_B]['history']['InitialStellarMass']])

        formationTime = np.hstack([pickle['data'][key_A]['history']['formationTime'], \
                                   pickle['data'][key_B]['history']['formationTime']])
        
        selection = np.random.choice(InitialStellarMass.shape[0], \
                                     int(InitialStellarMass.shape[0]/2))
        
        pickle['data'][key]['history']['InitialStellarMass'] = InitialStellarMass[selection]
        pickle['data'][key]['history']['formationTime'] = formationTime[selection]
    
    return pickle


def bin_histories(pickle, binning='linear', name='linear', Nbins=20):

    # initialise bins
    if binning == 'linear':
        binLimits = np.linspace(0, cosmo.age(0).value, Nbins+1)
        binWidth = binLimits[1] - binLimits[0]
        bins = np.linspace(binWidth/2, binLimits[-1] - binWidth/2, Nbins)
        binWidths = binWidth * 1e9        
    elif binning == 'log':
        upperBin = np.log10(cosmo.age(0).value * 1e9)
        binLimits = np.linspace(7.1, upperBin, Nbins+1)
        binWidth = binLimits[1] - binLimits[0]
        bins = np.linspace(7.1 + binWidth/2., upperBin - binWidth/2., Nbins)
        binWidths = 10**binLimits[1:] - 10**binLimits[:len(binLimits)-1]
        binLimits = 10**binLimits / 1e9
    else:
        raise ValueError('Invalid binning chosen, use either \'linear\' or \'log\'')
        
    
    # save binning info to header
    pickle['header'][name] = {}
    pickle['header'][name]['binLimits'] = binLimits
    pickle['header'][name]['binWidth'] = binWidth
    pickle['header'][name]['bins'] = bins
    pickle['header'][name]['binWidths'] = binWidths
    
    
    # create a lookup table of ages to avoid calculating for every particle
    age_lookup = np.linspace(1e-4, cosmo.age(0).value - 1e-2, 20000)
    z_lookup = [z_at_value(cosmo.lookback_time, a) for a in age_lookup * u.Gyr]

    for key, value in pickle['data'].items():

        particles = value['history']

        # calculate age in Gyr using lookup table
        formation_age = age_lookup[np.searchsorted(z_lookup, scale_factor_to_z(particles['formationTime']))]

        # # calculate age in Gyr directly
        # formation_age = cosmo.lookback_time(scale_factor_to_z(formation_time))

        ## Linear Bin
        counts, dummy = np.histogram(formation_age, 
                bins=binLimits, weights=particles['InitialStellarMass']);  # weights converts age to SFR

        value[name] = {}
        value[name]['SFH'] = counts / binWidths # divide by bin width in (Giga)years to give SFR in Msol / year

    
    return pickle

