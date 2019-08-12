import requests
import numpy as np
import h5py

from scipy.spatial.distance import cdist

def get(path, params=None, filename=None, keyfile='api-key.txt'):
    """
    Query Illustris API

    Args:
    path (str): API query path
    params: ...
    filename (str): filename and directory to save content.
    """
    
    with open('api-key.txt', 'r') as myfile:
        key=myfile.read()

    headers = {"api-key": key};

    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers);

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status();

    if r.headers['content-type'] == 'application/json':
        return r.json(); # parse json responses automatically

    if 'content-disposition' in r.headers:

        if filename is None:
            filename = 'data/'+r.headers['content-disposition'].split("filename=")[1];

        with open(filename, 'wb') as f:
            f.write(r.content);

        return filename; # return the filename string

    return r;


def get_sub_url(sub_id, sim='Illustris-1', snap='135'):
    """
    Get URL for given subhalo

    Args:
    sub_id (int): id number for subhalo
    sim (str): chosen simulation
    snap (str): chosen snapshot number

    Returns:
    sub_url (str) url string for subhalo query
    """
    # base_url = "http://www.illustris-project.org/api/"
    base_url = "http://www.tng-project.org/api/"
    
    sim_properties = get(base_url+sim)
    return "%s%s/snapshots/%s/subhalos/%s/"%(base_url, sim, snap, sub_id)


def get_sub_spectra(sub_id, sim, snap, verbose=False):
    """
    Get subhalo spectra

    Args:
    sub_id (int): id number for subhalo
    sim (str): chosen simulation
    snap (str): chosen snapshot number

    Returns:
    spec (array, float): 1D spectra
    """


    sub_url = get_sub_url(sub_id, sim, snap)

    spectra_url = "%sstellar_mocks/sed.txt"%(sub_url)
    spectra = get(spectra_url).text

    spec = np.array([x.split(',') for x in spectra.split('\n')]) # numpy array

    if verbose == True: print(spec[0])

    spec = spec[1:].astype('float')

    return spec


def get_sub_particles(sub_id, sim='Illustris-1', snap='135', mask_wind=True, verbose=False, distance=30):
    """
    Get subhalo particle cutout information.

    Args:
    sub_id (int): id number for subhalo
    sim (str): chosen simulation
    snap (str): chosen snapshot number
    mask_wind (bool): mask wind particles, defined as those with formation ages < 0
    verbose (bool)

    Returns:
    particles (dict)(array, float}: particle information arrays in labelled dict
    """

    sub_url = get_sub_url(sub_id, sim, snap)
    print(sub_url)

    meta = get(sub_url)['meta']
    h = get(meta['simulation'])['hubble']
    scale_factor = 1 / (1+get(meta['snapshot'])['redshift'])
    
    # Find centre of mass
    sh = get(sub_url)
    subhalo_com = (np.expand_dims([sh['pos_x'], sh['pos_y'], sh['pos_z']], 0) / h) * scale_factor # physical kpc

    # SFH
    cutout_request = {'stars':'GFM_StellarFormationTime,GFM_InitialMass,GFM_Metallicity,Coordinates',
                      'gas':'StarFormationRate,Masses'};

    # use same filename to avoid storing hundreds of hdf5 files
    cutout = get(sub_url+"cutout.hdf5", cutout_request, filename='data/temp.hdf5');
    
    particles = {}

    with h5py.File(cutout,'r') as f:
        
        coods = (f['PartType4']['Coordinates'][:] / h) * scale_factor  # convert to physical units
        
        p_mask = cdist(subhalo_com, coods)[0] < distance
        
        # particles['coods'] = coods[p_mask]
        particles['formationTime'] = f['PartType4']['GFM_StellarFormationTime'][p_mask]; # scale factor
        particles['InitialStellarMass'] = f['PartType4']['GFM_InitialMass'][p_mask] * 1e10 / h; # Msol
        particles['Metallicity'] = f['PartType4']['GFM_Metallicity'][p_mask]
        
        if len(f['PartType0']) > 0:
            star_forming_gas_mask = f['PartType0']['StarFormationRate'][:] > 0.
            star_forming_gas_mass = np.sum(f['PartType0']['Masses'][:][star_forming_gas_mask] * 1e10 / h) 
        else:
            star_forming_gas_mass = 0.


    if mask_wind:  # mask out wind particles
        mask = particles['formationTime'] > 0.

        # particles['coods'] = particles['coods'][mask]
        particles['formationTime'] = particles['formationTime'][mask]
        particles['InitialStellarMass'] = particles['InitialStellarMass'][mask]
        particles['Metallicity'] = particles['Metallicity'][mask]

    return particles, star_forming_gas_mass, subhalo_com



def get_sub_particles_fiber(sub_id, sim='Illustris-1', snap='135', mask_wind=True, verbose=False, distance=3, axis=0):
    """
    Get subhalo particle cutout information.

    Args:
    sub_id (int): id number for subhalo
    sim (str): chosen simulation
    snap (str): chosen snapshot number
    mask_wind (bool): mask wind particles, defined as those with formation ages < 0
    verbose (bool)

    Returns:
    particles (dict)(array, float}: particle information arrays in labelled dict
    """

    sub_url = get_sub_url(sub_id, sim, snap)
    print(sub_url)

    meta = get(sub_url)['meta']
    h = get(meta['simulation'])['hubble']
    scale_factor = 1 / (1+get(meta['snapshot'])['redshift'])
    
    # Find centre of mass
    sh = get(sub_url)
    subhalo_com = (np.expand_dims([sh['pos_x'], sh['pos_y'], sh['pos_z']], 0) / h) * scale_factor # physical kpc

    # SFH
    cutout_request = {'stars':'GFM_StellarFormationTime,GFM_InitialMass,GFM_Metallicity,Coordinates',
                      'gas':'StarFormationRate,Masses'};

    # use same filename to avoid storing hundreds of hdf5 files
    cutout = get(sub_url+"cutout.hdf5", cutout_request, filename='data/temp.hdf5');
    
    particles = {}

    with h5py.File(cutout,'r') as f:
        
        coods = (f['PartType4']['Coordinates'][:] / h) * scale_factor  # convert to physical units

        axes = np.delete(np.arange(3), axis)
        p_mask = cdist([subhalo_com[0][axes]], coods[:,axes])[0] < distance

        # randomly select some fraction of particles beyond `distance`
        # selection = np.random.choice(np.where(~p_mask)[0], int(len(p_mask) * fraction))
        # p_mask[selection] = True

        # particles['coods'] = coods[p_mask] 
        particles['formationTime'] = f['PartType4']['GFM_StellarFormationTime'][p_mask]; # scale factor
        particles['InitialStellarMass'] = f['PartType4']['GFM_InitialMass'][p_mask] * 1e10 / h; # Msol
        particles['Metallicity'] = f['PartType4']['GFM_Metallicity'][p_mask]
        
        if len(f['PartType0']) > 0:
            star_forming_gas_mask = f['PartType0']['StarFormationRate'][:] > 0.
            star_forming_gas_mass = np.sum(f['PartType0']['Masses'][:][star_forming_gas_mask] * 1e10 / h) 
        else:
            star_forming_gas_mass = 0.


    if mask_wind:  # mask out wind particles
        mask = particles['formationTime'] > 0.

        # particles['coods'] = particles['coods'][mask]
        particles['formationTime'] = particles['formationTime'][mask]
        particles['InitialStellarMass'] = particles['InitialStellarMass'][mask]
        particles['Metallicity'] = particles['Metallicity'][mask]

    return particles, star_forming_gas_mass, subhalo_com
