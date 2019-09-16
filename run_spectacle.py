import numpy as np
import h5py
import schwimmbad
from functools import partial

# from spectacle import spectacle
from predict import predict

def main(pool):
    
    # choose filename
    fname = 'data/full_histories_eagle.h5'#'data/full_histories_illustris.h5'
    # SFR timescales?
    A = False
    # Bin histories?
    B = False
    # Generate spectra?
    C = True
    # grid name?
    grid_name = 'fsps_neb'
    # dust attenuated spectra?
    D = True

    # Load data
    tacle = predict(fname=fname)

    ## Load subhalo IDs for use in pool calls
    shids = tacle.load_arr('ID','Subhalos')
    print("Number of subhalos: %i"%len(shids))

    if A:
        ## SFR timescales (in parallel) ##
        print("Calculating SFR timescales...")
        print('SFR 100:')
        lg = partial(tacle.recalculate_sfr, time=0.1)
        sfr100 = np.array(list(pool.map(lg,shids)))
        
        print('SFR 10:')
        lg = partial(tacle.recalculate_sfr, time=0.01)
        sfr10 = np.array(list(pool.map(lg,shids)))
        
        print('SFR 50:')
        lg = partial(tacle.recalculate_sfr, time=0.05)
        sfr50 = np.array(list(pool.map(lg,shids)))
        
        print('SFR 500:')
        lg = partial(tacle.recalculate_sfr, time=0.5)
        sfr500 = np.array(list(pool.map(lg,shids)))
        
        print('SFR 1000:')
        lg = partial(tacle.recalculate_sfr, time=1.0)
        sfr1000 = np.array(list(pool.map(lg,shids)))
    
        tacle.save_arr(sfr100,'SFR 100Myr','Subhalos')
        tacle.save_arr(sfr10,'SFR 10Myr','Subhalos')
        tacle.save_arr(sfr50,'SFR 50Myr','Subhalos')
        tacle.save_arr(sfr500,'SFR 500Myr','Subhalos')
        tacle.save_arr(sfr1000,'SFR 1Gyr','Subhalos')


    if B:
        ## Bin histories in parallel ##
        print("Binning star formation histories...")
        tacle.delete_arr('SFH','')
     
        upperBin = np.log10(tacle.cosmo.age(tacle.redshift).value * 1e9)
        binLimits = np.hstack([[0.0], np.linspace(7.5, 9.5, num=7), upperBin])
        binWidths = 10**binLimits[1:] - 10**binLimits[:len(binLimits)-1]
        bins = binLimits[:-1] + ((binLimits[1:] - binLimits[:-1]) / 2)
        binLimits = 10**binLimits / 1e9
        custom = {'binLimits': binLimits, 'bins': bins, 'binWidths': binWidths}
    
        bins, binLimits, binWidths = tacle.init_bins(name='log_8', custom=custom, verbose=True)
    
        lg = partial(tacle.bin_histories, binLimits=binLimits, binWidths=binWidths)
        sfh = np.array(list(pool.map(lg,shids)))
    
        tacle.save_arr(sfh,'log_8','SFH')

    if C:
        tacle._clear_spectra()
        print("Generating spectra...")
        grid = tacle.load_grid(name=grid_name, grid_directory=tacle.grid_directory)
        grid = tacle.redshift_grid(grid,tacle.redshift)
        Z = grid['metallicity']
        A = grid['age'][tacle.redshift]
        wl = grid['wavelength']
        
        resample_wl = np.loadtxt('data/wavelength_grid.txt')
        # tacle.create_lookup_table(tacle.redshift)
        tacle.load_lookup_table(tacle.redshift)
    
        ## Calculate weights (in parallel)
        lg = partial(tacle.weights_grid, Z=Z, A=A, resample=True, verbose=True)
        weights = np.array(list(pool.map(lg,shids)))
    
        intrinsic_spectra = tacle.calc_intrinsic_spectra(weights,grid,z=tacle.redshift)
        intrinsic_spectra = tacle.rebin_spectra(intrinsic_spectra, grid['wavelength'],resample_wl)
        tacle.save_spectra('Intrinsic', intrinsic_spectra, resample_wl, units=str('Lsol / AA'))
    
        intrinsic_spectra = tacle.add_noise_flat(intrinsic_spectra, resample_wl)
        tacle.save_spectra('Noisified Intrinsic', intrinsic_spectra, resample_wl, units=str('Lsol / AA'))
    
        M_g = tacle.calculate_photometry(intrinsic_spectra, resample_wl, filter_name='SDSS_g')
        M_r = tacle.calculate_photometry(intrinsic_spectra, resample_wl, filter_name='SDSS_r')
        tacle.save_arr(M_g,'M_g','Photometry/Intrinsic/')
        tacle.save_arr(M_g,'M_g','Photometry/Intrinsic/')

        if D:
            ## Dust metallicity factor
            shids = tacle.load_arr('ID','Subhalos')
            sfgmass = tacle.load_arr('Star Forming Gas Mass','Subhalos')
            smass = tacle.load_arr('Stellar Mass','Subhalos')
            sfgmet = tacle.load_arr('Gas Metallicity','Subhalos')
        
            tau_0 = tacle.tau_0()
            print("tau_0:",tau_0)
            Z_factor = tacle.metallicity_factor_update(sfgmet,sfgmass,smass,
                                                       gas_norm=1,metal_norm=1,tau_0=tau_0)
    
            dust_spectra = tacle.two_component_dust(weights,grid,z=tacle.redshift,
                                                    tau_ism=0.33 * Z_factor, tau_cloud=0.67 * Z_factor)
    
            dust_spectra = tacle.rebin_spectra(dust_spectra, grid['wavelength'],resample_wl)
            tacle.save_spectra('Dust', dust_spectra, resample_wl, units=str('Lsol / AA'))
    
            dust_spectra = tacle.add_noise_flat(dust_spectra, resample_wl)
            tacle.save_spectra('Noisified Dust', dust_spectra, resample_wl, units=str('Lsol / AA'))

            M_g = tacle.calculate_photometry(dust_spectra, resample_wl, filter_name='SDSS_g')
            M_r = tacle.calculate_photometry(dust_spectra, resample_wl, filter_name='SDSS_r')
            tacle.save_arr(M_g,'M_g','Photometry/Dust/')
            tacle.save_arr(M_r,'M_r','Photometry/Dust/')


    pool.close()



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Spectacle")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    print(pool)
    main(pool)

    print("All done. Spec-tacular!")

