import numpy as np

w_min_val = 3572.0
w_max_val = 8173.0

new_wl = 10**np.linspace(np.log10(w_min_val), np.log10(w_max_val), w_max_val - w_min_val)
np.savetxt('data/wavelength_grid.txt', new_wl)
    
