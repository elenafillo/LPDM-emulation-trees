import numpy as np
import xarray as xr
import glob
import dask

"""
Reproducible code for "A machine learning emulator for Lagrangian particle dispersion model footprints: A case study using NAME" 
by Elena Fillola, Raul Santos-Rodriguez, Alistair Manning, Simon O'Doherty and Matt Rigby (2022)

Author: Elena Fillola (elena.fillolamayoral@bristol.ac.uk)
"""

class LoadData:
    """
    Load data for training and testing, for a particular site

    inputs:
        - year: can be an int (eg 2016) or a string, including combinations of years (eg "2016", "201[4-5]")
        - site: site identifyer, as a string. Default is Mace Head ("MHD")
        - siteheight: height of footprints as a string of numbers (eg "100magl"). If empty, default height is used.
        - metheight: height for met data as a string of numbers (eg "100magl"). Default is 10magl.
        - size: size for footprint to be cut to, as an int. Resolution of initial footprint is maintained, cut to a sizexsize square around the release point. 
            Should be even for computational purposes.
        - verbose: if True, prints out the steps throughout the data loading process.
        - met_datadir: Directory for .nc met data as a string, including wildcards if needed (eg "data/MHD_*"). If empty, uses default folder and naming. 
        - extramet_datadir: Directory for .nc extramet data (used for gradients, needs to be preprocessed to have some time and space resolution as met) as a string. 
            including wildcards if needed (eg "data/MHD_*"). If empty, uses default folder and naming.
        - fp_datadir: Directory for .nc footprint data as a string, including wildcards if needed (eg "data/MHD_*"). If empty, uses default folder and naming.
        Note for all three directories: If not empty, only appends year (ie need to specify or use wildcards for rest of name, including site or domain) 
        If passing wildcards from command line, need to do so in quotes.
    outputs:
        LoadData object with attributes:
        - year, site, metheight, size, metsize: details about inputs
        - met: meteorology input files, cut to size
        - fp_data_full: footprint input files
        - fp_data: flattened np array with footprint cut to size, with shape (n_samples, size**2)
        - release_lat, release_lon: coordinates of site
        - fp_lats, fp_lons: latitudes and longitudes for each cell in the cut footprint
        - temp_grad, x_wind_grad, y_wind_grad: vertical gradients extracted from extramet input files, each as 
            a flattened np array with shape (n_samples, size**2). Note last three items (timewise) are nan due to interpolation
        - y_wind and x_wind: horizontal wind vectors, transformed from input data's wind direction and speed.

    """
    def __init__(self, year, site = "MHD", domain=None, siteheight = None, metheight = None, size=10, verbose = False,
        met_datadir = None, extramet_datadir = None, fp_datadir = None):
        heights = {"MHD":"10magl", "THD":"10magl", "TAC":"185magl", "RGL":"90magl", "HFD":"100magl", "BSD":"250magl", "GSN":"10magl"} # default heights
        if siteheight != None:
            heights[site] = siteheight
        if siteheight == None:
            assert site in heights, "There is no default height for this site. Pass siteheight input (as a string of numbers)"
        
        if domain==None:
            domains = {"MHD":"EUROPE", "THD":"USA", "TAC":"EUROPE", "RGL":"EUROPE", "HFD":"EUROPE", "BSD":"EUROPE", "GSN":"EASTASIA"}
            try:
                domain = domains[site]   
            except: 
                print("No domain was passed and there is no default domain for this site.")
                if met_datadir != None and extramet_datadir != None: 
                    print("Domain is not needed because met_datadir and extramet_datadir were passed")
                else:    
                    print("If domain is not passed custom paths for met_datadir and extramet_datadir should be passed.")
                    domain=""
                   
        ## TODO remove all mentions of this and fix directly
        self.year = year
        self.site = site
        if metheight==None:
            self.metheight = "10magl"
        else:
            self.metheight = metheight

        ## load met data
        
        if met_datadir==None:
            met_datadir = "data/met/"+str(domain)+"*"+str(self.metheight)+"*"+str(self.year)+"*.nc"
        else:
            met_datadir = met_datadir+str(self.year)+"*.nc"
        if verbose: print("Loading Meteorology data from " + met_datadir)
        self.met = xr.open_mfdataset(sorted(glob.glob(met_datadir)), combine='by_coords')
        
        self.size = size
        self.metsize = size+6

        ## check if any of the met entries are nans, and if so remove from met
        if np.sum(np.isnan(self.met.PBLH.values)) != 0:
            nan_idxs = np.unique(np.where(np.isnan(self.met.PBLH.values))[2])
            self.met = self.met.sel(time=np.delete(self.met.time.values, nan_idxs))

        # load footprint (fp) data
        if fp_datadir==None:
            fp_datadir = "data/fps/"+site+"-"+heights[site]+"*"+str(self.year)+"*.nc"
        else:
            fp_datadir=fp_datadir+str(self.year)+"*.nc"
        if verbose: print("Loading footprint data from " + fp_datadir) 
        self.fp_data_full = xr.open_mfdataset(sorted(glob.glob(fp_datadir)), combine='by_coords')

        self.release_lat = self.fp_data_full.release_lat[0].values
        self.release_lon = self.fp_data_full.release_lon[0].values

        self.fp_data_full= self.fp_data_full.sortby('time')
        self.met= self.met.sortby('time')

        ## checking that the domain is the same
        assert np.sum(self.met.lat != self.fp_data_full.lat) == 0, "it looks like the met and the fps are in different domains!"

        ## check that met and fp data are the same length
        if len(self.met.time.values) != len(self.fp_data_full.time.values):
            print("There's not the same temporal data for met and fp so cutting them both to match ("+str(len(self.met.time.values))+" time values for met data, "+str(len(self.fp_data_full.time.values))+" time values for fp data)")

            intersection = list(set(self.met.time.values).intersection(self.fp_data_full.time.values)) 
            self.met = self.met.sel(time=intersection)
            self.fp_data_full = self.fp_data_full.sel(time=intersection)

        ## cut both to size around release point
        if verbose: print("Cutting data to size")
        self.met = cut_met(self.met, self.release_lat, self.release_lon, self.metsize)
        self.fp_data, self.fp_lats, self.fp_lons = cut_data(self.fp_data_full, self.release_lat, self.release_lon, self.size, returnlatlons = True)
        


        ## load extra meteorology, interpolate and extract gradients
        ### note that the extra meteorology has been preprocessed, with two levels selected and time/space interpolated
        ### note as data is 3-hourly the last three values cannot be interpolated 
        if extramet_datadir==None:
            extramet_datadir = "data/met/"+str(domain)+"*verticalgradients*"+str(self.year)+"*.nc"
        else:
            extramet_datadir = extramet_datadir+str(self.year)+"*.nc"
        if verbose: print("Loading extra meteorology from " + extramet_datadir + " and extracting gradients")

        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            extramet = xr.open_mfdataset(sorted(glob.glob(extramet_datadir)), combine='by_coords')

            extramet = extramet.sel(latitude = self.met.lat.values, longitude = self.met.lon.values, method="nearest")
            extramet = extramet.interp(time = self.met.time.values)        
            extramet = extramet.interpolate_na()

        ## save the tree gradients as a flattened np array (n_samples, size**2)
        self.temp_grad = extramet.air_temperature.sel(model_level_number=4) - extramet.air_temperature.sel(model_level_number=1)
        self.temp_grad = np.reshape(self.temp_grad.values, (len(self.temp_grad.time), (self.metsize)**2))
        self.x_wind_grad = extramet.x_wind.sel(model_level_number=4) - extramet.x_wind.sel(model_level_number=1)
        self.x_wind_grad = np.reshape(self.x_wind_grad.values, (len(self.x_wind_grad.time), (self.metsize)**2))
        self.y_wind_grad = extramet.y_wind.sel(model_level_number=4) - extramet.y_wind.sel(model_level_number=1)
        self.y_wind_grad = np.reshape(self.y_wind_grad.values, (len(self.y_wind_grad.time), (self.metsize)**2))

        ## transform wind from speed and direction to x and y vectors
        if verbose: print("Extracting wind vectors")
        self.y_wind, self.x_wind = generate_wind_vectors(self.met.Wind_Speed, self.met.Wind_Direction)

        if verbose: print("All data loaded")


def cut_met(met, release_lat, release_lon, size):
    ## cuts meteorology to size around release point and returns as a smaller xarray
    release_lat, release_lon = min(met.lat.values, key=lambda x:abs(x-release_lat)), min(met.lon.values, key=lambda x:abs(x-release_lon))
    idx_release_lat = np.where(met.lat.values == release_lat)[0][0]
    idx_release_lon = np.where(met.lon.values == release_lon)[0][0]
    half = int(size/2)
    lats = met.lat.values[idx_release_lat-half:idx_release_lat+half]
    lons = met.lon.values[idx_release_lon-half:idx_release_lon+half]
    met = met.sel({"lat":lats, "lon":lons}).compute()
    return met

def cut_data(fp_full, release_lat, release_lon, size, returnlatlons = False):
    ## cuts footprint to size around release point and returns as a flattened np array (n_samples, size**2)
    release_lat, release_lon = min(fp_full.lat.values, key=lambda x:abs(x-release_lat)), min(fp_full.lon.values, key=lambda x:abs(x-release_lon))
    idx_release_lat = np.where(fp_full.lat.values == release_lat)[0][0]
    idx_release_lon = np.where(fp_full.lon.values == release_lon)[0][0]    
    half = int(size/2)
    lats = fp_full.lat.values[idx_release_lat-half:idx_release_lat+half]
    lons = fp_full.lon.values[idx_release_lon-half:idx_release_lon+half]
    
    data = fp_full.sel({"lat":lats, "lon":lons}).fp.values

    data = data.reshape((np.shape(data)[0]*np.shape(data)[1], np.shape(data)[2]))
    data = np.transpose(data, [1,0])

        
    if returnlatlons:
        return data, lats, lons
    else:
        return data      


def generate_wind_vectors(windspeed, winddir):
    y_wind = windspeed.values * np.cos(np.radians(winddir.values))
    x_wind = windspeed.values * np.sin(np.radians(winddir.values))   
    return y_wind, x_wind

def get_past_values(met_var, jump):
    """
    Concatenate a variable and its past.

    Returns an np array of shape (n_samples-jump, 2*metsize**2). 
    Second dimension is concatenation of [variable at fp time (size metsize**2), variable #jump hours before (size metsize**2)].
    Takes as inputs a variable (as a np array of shape (metsize, metsize, n_samples)) and a "jump" in hours as an int
    eg get_past_values(data.u_wind, 6) will return the u_wind at the time of the footprint and six hours before.
    """
    met_var = np.transpose(met_var, [2,0,1])
    met_var = np.reshape(met_var, (np.shape(met_var)[0], np.shape(met_var)[1]**2))

    return np.concatenate((met_var[jump:,:], met_var[:-jump,:]), axis=1)


def get_all_inputs(variables_past, jump, variables_nopast):
    """
    Returns all input variables stacked and ready to pass to regressors, with shape (n_samples-jump-3, (2*#variables_past+#variables_nopast)**2)

    Takes a list of variables that are passed at time of footprint and #jump hours before (in paper these are x_wind, y_wind, PBLH with jump=6)
    and a list of variables that are passed only at time of footprint. 
    Last three items of all variables are removed, due to interpolation setup.
    """
    all_vars = []
    for v in variables_past:
        all_vars.append(get_past_values(v, jump)[:-3, :])

    for v in variables_nopast:
        all_vars.append(v[jump:-3, :])
    
    return np.hstack(all_vars)