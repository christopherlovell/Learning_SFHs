
import numpy as np

def scale_factor_to_z(a):
    return (1./a) - 1


def norm_spectra(lamb, L, norm_wl = 1.6e-6):
    """
    Normalise a spectra to the flux at a given wavelength
    """
    
    idx = np.argmin(lamb < norm_wl)
    return L / L[idx]


def burstiness(sfh, tol = 1.):
    """
    Evaluates the number of burst periods in a given binned SFH. Evaluates absolute ratio of change in sfr between two bins, allowing bursts and sudden quiescence periods to be captured.
    
    Args:
    sfh (array, float) - binned star formation history
    tol (float) - delta SF tolerance, above which a burst is declared
    
    Returns:
    (array, bool) shape len(sfh) - 1, indicating burst events
    """
    
    delta = np.diff(pred[::-1])  # increase in SFR between consecutive bins
    s_minus_1 = pred[::-1][:-1]  # SFR in previous bin
    
    burst = np.abs(delta / s_minus_1)  # absolute ratio of sfr change
    
    return burst > tol  # burst events above tolerance threshold
