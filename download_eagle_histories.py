import eagle_IO.eagle_IO as E
import numpy as np
import h5py
from scipy.spatial.distance import cdist


directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'
tag = '027_z000p101'


p_sgrpn = E.read_array('PARTDATA', directory, tag, '/PartType4/SubGroupNumber',noH=True, physicalUnits=False,numThreads=1)
p_grpn = E.read_array('PARTDATA', directory, tag, '/PartType4/GroupNumber',noH=True, physicalUnits=False,numThreads=1)
p_imass = E.read_array('PARTDATA', directory, tag, '/PartType4/InitialMass',noH=True, physicalUnits=False,numThreads=1) * 1e10
p_form = E.read_array('PARTDATA', directory, tag, '/PartType4/StellarFormationTime',noH=True, physicalUnits=False,numThreads=1)
p_metal = E.read_array('PARTDATA', directory, tag, '/PartType4/SmoothedMetallicity',noH=True, physicalUnits=False,numThreads=1)
p_cood = E.read_array('PARTDATA', directory, tag, '/PartType4/Coordinates',noH=True, physicalUnits=False,numThreads=1)

shgrpno = E.read_array('SUBFIND', directory, tag, '/Subhalo/SubGroupNumber',noH=True, physicalUnits=False,numThreads=1)
grpno = E.read_array('SUBFIND', directory, tag, '/Subhalo/GroupNumber',noH=True, physicalUnits=False,numThreads=1)

mstar_30 = E.read_array('SUBFIND', directory, tag, '/Subhalo/ApertureMeasurements/Mass/030kpc',noH=True, physicalUnits=False,numThreads=1)[:,4] * 1e10
mstar = E.read_array('SUBFIND', directory, tag, '/Subhalo/Stars/Mass',noH=True, physicalUnits=False,numThreads=1) * 1e10

stellar_metallicity = E.read_array('SUBFIND', directory, tag, '/Subhalo/Stars/Metallicity',noH=True, physicalUnits=False,numThreads=1)
sf_metal = E.read_array('SUBFIND', directory, tag, '/Subhalo/SF/Metallicity',noH=True, physicalUnits=False,numThreads=1)
sf_mass = E.read_array('SUBFIND', directory, tag, '/Subhalo/SF/Mass',noH=True, physicalUnits=False,numThreads=1) * 1e10
sfr_30 = E.read_array('SUBFIND', directory, tag, '/Subhalo/ApertureMeasurements/SFR/030kpc',noH=True, physicalUnits=False,numThreads=1)
sfr = E.read_array('SUBFIND', directory, tag, '/Subhalo/StarFormationRate',noH=True, physicalUnits=False,numThreads=1)
cop = E.read_array('SUBFIND', directory, tag, '/Subhalo/CentreOfPotential',noH=True, physicalUnits=False,numThreads=1)


log_masses = np.log10(mstar)
indices = np.where(log_masses > 10.0)[0]
shape = len(indices)
print("N: %s"%len(indices))


with h5py.File('data/full_histories_eagle.h5','a') as f:

    if "Star Particles" not in list(f.keys()): f.create_group("Star Particles")
    if "Subhalos" not in list(f.keys()): f.create_group("Subhalos")

    ## delete fields ##
    for field in list(f["Subhalos"].keys()):
        del f['Subhalos/%s'%field]
          
    for field in list(f["Star Particles"].keys()):
        del f['Star Particles/%s'%field] 


with h5py.File('data/full_histories_eagle.h5','a') as f:
    
    if "ID" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('ID',(shape,),dtype='i8',maxshape=(None,),data=np.ones(shape)*-1)
    
    if "Stellar Mass" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Stellar Mass',(shape,),maxshape=(None,))
        
    if "Gas Metallicity" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Gas Metallicity',(shape,),maxshape=(None,))
        
    if "Stellar Metallicity" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Stellar Metallicity',(shape,),maxshape=(None,))
        
        f["Subhalos"].create_dataset('SFR',(shape,),maxshape=(None,))
        
    if "Star Forming Gas Mass" not in list(f["Subhalos"].keys()):
        f['Subhalos'].create_dataset('Star Forming Gas Mass',(shape,),maxshape=(None,))
        
    if "Stellar Mass 30kpc" not in list(f["Subhalos"].keys()):
        f['Subhalos'].create_dataset('Stellar Mass 30kpc',(shape,),maxshape=(None,))
        
    if "Stellar Metallicity 30kpc" not in list(f["Subhalos"].keys()):
        f['Subhalos'].create_dataset('Stellar Metallicity 30kpc',(shape,),maxshape=(None,))
        
    if "Index Start" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Index Start',(shape,),dtype='i8',maxshape=(None,))
        
    if "Index Length" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Index Length',(shape,),dtype='i8',maxshape=(None,))
    
    if "Formation Time" not in list(f["Star Particles"].keys()):
        f["Star Particles"].create_dataset('Formation Time',(0,),maxshape=(None,))
        
    if "Initial Mass" not in list(f["Star Particles"].keys()):
        f["Star Particles"].create_dataset('Initial Mass',(0,),maxshape=(None,))
        
    if "Metallicity" not in list(f["Star Particles"].keys()):
        f["Star Particles"].create_dataset('Metallicity',(0,),maxshape=(None,))
        

with h5py.File('data/full_histories_eagle.h5','a') as f:

    pidx = len(f["Star Particles/Formation Time"][:])
    print("Initial pidx:", pidx)

    for i,key in enumerate(indices):
        # print(key)

        mask = (p_sgrpn == shgrpno[key]) & (p_grpn == grpno[key])
        mask[mask] = cdist(p_cood[mask], np.expand_dims(cop[key], 0))[:,0] < 0.03

        if key in f["Subhalos/ID"][:]:
            continue

        if (i % 100) == 0: print(round(float(i)/shape * 100,2), "%")

        
        f["Subhalos/ID"][i] = key
        f["Subhalos/Stellar Mass"][i] = mstar[key]
        f["Subhalos/Gas Metallicity"][i] = sf_metal[key]
        f["Subhalos/SFR"][i] = sfr[key]
        f["Subhalos/Star Forming Gas Mass"][i] = sf_mass[key]
        f["Subhalos/Stellar Mass 30kpc"][i] = mstar_30[key] 

        plen = np.sum(mask) # len(particles['formationTime'])
        # print(pidx,plen,pidx+plen)
        
        f["Subhalos/Index Start"][i] = pidx
        f["Subhalos/Index Length"][i] = plen
        
        dset = f["Star Particles/Formation Time"]
        dset.resize(size=(pidx+plen,))
        dset[pidx:] = p_form[mask]
        
        dset = f["Star Particles/Initial Mass"]
        dset.resize(size=(pidx+plen,))
        dset[pidx:] = p_imass[mask]
        
        dset = f["Star Particles/Metallicity"]
        dset.resize(size=(pidx+plen,))
        dset[pidx:] = p_metal[mask]
        
        pidx += plen
        
        f.flush()

    
