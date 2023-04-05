#I AM HUMAN
# I AM ROBOT
# I AM GAIA
import xarray as xr
import numpy as np
import statsmodels.api as sm
import pandas as pd
import json 
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
from sklearn import linear_model
import glob
from scipy import signal
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import matplotlib as mpl
import random
# To use R packages:
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
# Convert pandas.DataFrames to R dataframes automatically.
pandas2ri.activate()
relaimpo = importr("relaimpo")
    
#Across models regression class
class spatial_MLR(object):
    def __init__(self):
        self.what_is_this = 'This performs a regression across models and plots everything'
    
    def regression_data(self,variable):
        """Define the regression target variable 
        this is here to be edited if some opperation is needed on the DataArray
        
        :param variable: DataArray
        :return: target variable for the regression  
        """
        self.target = variable

    #Regresion lineal
    def linear_regression(self,x):
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        returns = [res.params[i] for i in range(self.rd_num)]
        return tuple(returns)

    def linear_regression_pvalues(self,x):
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        returns = [res.pvalues[i] for i in range(self.rd_num)]
        return tuple(returns)
    
    def linear_regression_R2(self,x):
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        return res.rsquared
    
    def linear_regression_relative_importance(self,x):
        y = self.regressors
        try:
            res = robjects.globalenv['rel_importance'](x,y)
            returns = [res[i] for i in range((len(res)))]
            #print(res)
            correct = True
        except:
            returns2 = [np.array([0.0]) for i in range(self.rd_num-1)]
            correct = False
        finally:
            if correct:
                #print('I should return ',returns)
                return tuple(returns)
            else:
                #print('I am returning 0')
                return tuple(returns2)

    def definitions(self,regressor_names): 
        """ Saves some class variables in case you just want to plot
        
        :param regressors: list with independent variables
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        
        #regressors  es un dataframe
        #rd_names es una lista de nombres para los archivos ['Aij','VBij',...]
        self.rd_num = len(regressor_names)
        self.regressor_names = regressor_names

        
    def perform_regression(self,regressors,regressor_names,path): 
        """ Performs regression over all gridpoints in a map and returns and saves DataFrames
        
        :param regressors: list with independent variables
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        
        #regressors  es un dataframe
        #rd_names es una lista de nombres para los archivos ['Aij','VBij',...]
        self.rd_num = len(regressor_names)
        self.regressor_names = regressor_names
    
        #Regresion lineal

        self.regression_y = sm.add_constant(regressors.values)
        self.regressors = regressors.values
        target_var = xr.apply_ufunc(replace_nans_with_zero, self.target)
        results = xr.apply_ufunc(self.linear_regression,target_var,input_core_dims=[["time"]],
                                 output_core_dims=[[] for i in range(self.rd_num)],
                                 vectorize=True,
                                 dask="parallelized")
        results_pvalues = xr.apply_ufunc(self.linear_regression_pvalues,target_var,input_core_dims=[["time"]],
                                 output_core_dims=[[] for i in range(self.rd_num)],
                                 vectorize=True,
                                 dask="parallelized")
        results_R2 = xr.apply_ufunc(self.linear_regression_R2,target_var,input_core_dims=[["time"]],
                                 output_core_dims=[[]],
                                 vectorize=True,
                                 dask="parallelized")
        
        relative_importance = xr.apply_ufunc(self.linear_regression_relative_importance,target_var,input_core_dims=[["time"]],
                                 output_core_dims=[[] for i in range(self.rd_num-1)],
                                 vectorize=True,
                                 dask="parallelized")
      
        for i in range(self.rd_num):
            if i == 0:
                regression_coefs = results[0].to_dataset()
            else:
                regression_coefs[self.regressor_names[i]] = results[i]
                
        regression_coefs = regression_coefs.rename({'ua':self.regressor_names[0]})
        regression_coefs.to_netcdf(path+'/regression_coefficients.nc')
        
        for i in range(self.rd_num):
            if i == 0:
                regression_coefs_pvalues = results_pvalues[0].to_dataset()
            else:
                regression_coefs_pvalues[self.regressor_names[i]] = results[i]
                
        regression_coefs_pvalues = regression_coefs_pvalues.rename({'ua':self.regressor_names[0]})
        regression_coefs_pvalues.to_netcdf(path+'/regression_coefficients_pvalues.nc')
        
        for i in range(len(relative_importance)):
            if i == 0:
                relative_importance_values = relative_importance[0].to_dataset()
            else:
                relative_importance_values[self.regressor_names[1:][i]] = relative_importance[i]
                
        relative_importance_values = relative_importance_values.rename({'ua':self.regressor_names[1]})
        print(relative_importance_values.tos_cp.max())
        relative_importance_values.to_netcdf(path+'/regression_coefficients_relative_importance.nc')
        results_R2.to_netcdf(path+'/R2.nc')
                     
        
    def create_x(self,i,j,dato):
        """ For each gridpoint creates an array and standardizes it 

        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """    
        x = np.array([])
        for y in range(len(dato.time)):
            aux = dato.isel(time=y)
            x = np.append(x,aux[i-1,j-1].values)
        return stand(x)
     
    def open_regression_coef(self,path):
        """ Open regression coefficients and pvalues to plot

        :param path: saving path
        :return maps: list of list of coefficient maps
        :return maps_pval:  list of coefficient pvalues maps
        :return R2: map of fraction of variance
        """ 
        maps = []; maps_pval = []
        coef_maps = xr.open_dataset(path+'/regression_coefficients.nc')
        coef_pvalues = xr.open_dataset(path+'/regression_coefficients_pvalues.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/R2.nc')
        return maps, maps_pval, R2    

    def open_lmg_coef(self,path):
        """ Open regression coefficients and pvalues to plot

        :param path: saving path
        :return maps: list of list of coefficient maps
        :return maps_pval:  list of coefficient pvalues maps
        :return R2: map of fraction of variance
        """ 
        maps = []; maps_pval = []
        coef_maps = xr.open_dataset(path+'/regression_coefficients_relative_importance.nc')
        coef_pvalues = xr.open_dataset(path+'/regression_coefficients_pvalues.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names[1:]]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/R2.nc')
        return maps, maps_pval, R2    
    
    def plot_regression_lmg_map(self,path,var,alias,output_path):
        """ Plots figure with all of 

        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_lmg_coef(path+'/'+var+'/'+alias)
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        path_era = '/datos/ERA5/mon'
        u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2018'))
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection = ccrs.SouthPolarStereo(central_longitude=300)
        data_crs = ccrs.PlateCarree()
        for k in range(self.rd_num-1):
            lat = maps[k].lat
            lon = np.linspace(0,360,len(maps[k].lon))
            var_c, lon_c = add_cyclic_point(maps[k].values,lon)
            #SoutherHemisphere Stereographic
            ax = plt.subplot(3,3,k+1,projection=projection)
            ax.set_extent([0,359.9, -90, 0], crs=data_crs)
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
            clevels = np.arange(0,40,2)
            im=ax.contourf(lon_c, lat, var_c*100,clevels,transform=data_crs,cmap='OrRd',extend='both')
            cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[k+1].min() < 0.05: 
                levels = [maps_pval[k+1].min(),0.05,maps_pval[k+1].max()]
                ax.contourf(maps_pval[k+1].lon, lat, maps_pval[k+1].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[k+1].min() < 0.10:
                levels = [maps_pval[k+1].min(),0.10,maps_pval[k+1].max()]
                ax.contourf(maps_pval[k+1].lon, lat, maps_pval[k+1].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k+1]) 
            plt.title(self.regressor_names[k+1],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
            plt1_ax = plt.gca()
            left, bottom, width, height = plt1_ax.get_position().bounds
            colorbar_axes1 = fig_coef.add_axes([left+0.23, bottom, 0.01, height*0.6])
            cbar = fig_coef.colorbar(im, colorbar_axes1, orientation='vertical')
            cbar.set_label('relative importance',fontsize=14) #rotation = radianes
            cbar.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        plt.savefig(output_path+'/regression_coefficients_relative_importance_u850_'+alias,bbox_inches='tight')
        plt.clf

        return fig_coef


    def plot_regression_coef_map(self,path,var,alias,output_path):
        """ Plots figure with all of 

        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_regression_coef(path+'/'+var+'/'+alias)
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        path_era = '/datos/ERA5/mon'
        u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2018'))
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection = ccrs.SouthPolarStereo(central_longitude=300)
        data_crs = ccrs.PlateCarree()
        for k in range(self.rd_num):
            lat = maps[k].lat
            lon = np.linspace(0,360,len(maps[k].lon))
            var_c, lon_c = add_cyclic_point(maps[k].values,lon)
            #SoutherHemisphere Stereographic
            ax = plt.subplot(3,3,k+1,projection=projection)
            ax.set_extent([0,359.9, -90, 0], crs=data_crs)
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
            if k == 0:
                im=ax.contourf(lon_c, lat, var_c,transform=data_crs,cmap=cmapU850,extend='both')
            else:
                clevels = np.arange(-.6,.7,0.1)
                im=ax.contourf(lon_c, lat, var_c,clevels,transform=data_crs,cmap=cmapU850,extend='both')
            cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[k].min() < 0.05: 
                levels = [maps_pval[k].min(),0.05,maps_pval[k].max()]
                ax.contourf(maps_pval[k].lon, lat, maps_pval[k].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[k].min() < 0.10:
                levels = [maps_pval[k].min(),0.10,maps_pval[k].max()]
                ax.contourf(maps_pval[k].lon, lat, maps_pval[k].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k]) 
            plt.title(self.regressor_names[k],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
            plt1_ax = plt.gca()
            left, bottom, width, height = plt1_ax.get_position().bounds
            colorbar_axes1 = fig_coef.add_axes([left+0.23, bottom, 0.01, height*0.6])
            cbar = fig_coef.colorbar(im, colorbar_axes1, orientation='vertical')
            cbar.set_label('m s$^{-1}$ $\sigma$_{regressor}^{-1}$',fontsize=14) #rotation = radianes
            cbar.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        plt.savefig(output_path+'/regression_coefficients_u850_'+alias,bbox_inches='tight')
        plt.clf

        return fig_coef


def stand_detr(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return signal.detrend(anom)

def filtro(dato):
    """Apply a rolling mean of 5 years and remov the NaNs resulting bigining and end"""
    signal = dato - dato.rolling(time=10, center=True).mean()
    signal_out = signal.dropna('time', how='all')
    return signal_out
                          
def stand(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return anom

def replace_nans_with_zero(x):
    return np.where(np.isnan(x), random.random(), x)


def multiple_linear_regression(target,predictors):
    """Multiple linear regression to estimate the links in  Causal Effect Network
    target: np.array - time series of target variable
    predictors: pandas dataframe with predictor variables
    
    """
    y = predictors.apply(stand_detr_filtro,axis=0).values
    print(predictors.keys())
    res = sm.OLS(stand_detr(target),y).fit()
    coef_output = {var:round(res.params[i],10) for var,i in zip(predictors.keys(),range(len(predictors.keys())))}
    coef_pvalues = {var:round(res.pvalues[i],10) for var,i in zip(predictors.keys(),range(len(predictors.keys())))}
    return coef_output, coef_pvalues
    
def figure(target,predictors):
    fig = plt.figure()
    y = predictors.apply(stand_detr,axis=0).values
    for i in range(len(predictors.keys())):
        plt.plot(y[:,i])
    plt.plot(stand_detr(target))
    return fig

def iod(iod_e,iod_w):
    iod_index = iod_w - iod_e
    return iod_index

def jet_lat_strength(jet_data,lon1=140,lon2=295):
    jet_30_70 = jet_data.sel(lat=slice(-70,-30)).sel(lon=slice(lon1,lon2)).mean(dim='lon')
    lat = jet_30_70.lat
    jet_lat = (jet_30_70*lat).sum(dim='lat')/(jet_30_70).sum(dim='lat')
    strength = []
    for t,max_lat in zip(jet_data.time,jet_lat):
        strength.append(jet_data.sel(time=t).sel(lat=max_lat,method='nearest').sel(lon=slice(lon1,lon2)).mean(dim='lon'))
    jet_strength = np.array(strength)
    return np.array(jet_lat.values),jet_strength

#fit <- lm(x~1+y)
robjects.r('''
    rel_importance <- function(x,y) {
        covar <-cov(cbind(y,x))
        metrics <- calc.relimp(covar, type = "lmg")
        return(metrics$lmg)
    }
''')

def main(config):
    """Run the diagnostic."""
    cfg=get_cfg(os.path.join(config["run_dir"],"settings.yml"))
    print(cfg)
    meta = group_metadata(config["input_data"].values(), "alias")
    #print(f"\n\n\n{meta}")
    for alias, alias_list in meta.items():
        #print(f"Computing index regression for {alias}\n")
        ts_dict = {m["variable_group"]: filtro(xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1900','2022'))).values for m in alias_list if (m["variable_group"] != "ua850") & (m["variable_group"] != "tos_iod_e") & (m["variable_group"] != "tos_iod_w")}
        ts_dict["iod"] = iod([filtro(xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1900','2022'))).values for m in alias_list if m["variable_group"] == "tos_iod_e"][0],[filtro(xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1900','2022'))).values for m in alias_list if m["variable_group"] == "tos_iod_w"][0])
        target_wind = [xr.open_dataset(m["filename"])[m["short_name"]] for m in alias_list if m["variable_group"] == "ua850"]
        target_wind = target_wind[0].sel(time=slice('1900','2022')) if len(target_wind) == 1 else 0
        target_wind_filtrado = filtro(target_wind)
        print(f"Computing spatial regression for {alias}\n")
        MLR = spatial_MLR()
        MLR.regression_data(target_wind_filtrado)
        os.chdir(config["work_dir"])
        os.getcwd()
        os.makedirs("u850_1900_2099",exist_ok=True)
        os.chdir(config["work_dir"]+'/'+"u850_1900_2099")
        os.makedirs(alias,exist_ok=True)
        #for i in range(len(target_wind_filtrado.values[0,:,0])):
        #    for j in range(len(target_wind_filtrado.values[0,0,:])):
        #        try:
        #            print('anda i: ',i)
        #            print('anda j: ',j)
        #            res = robjects.globalenv['rel_importance'](xr.apply_ufunc(replace_nans_with_zero, target_wind_filtrado).values[:,i,j],pd.DataFrame(ts_dict).apply(stand,axis=0).values)
        #            print(res)
        #        except:
        #            print('no anda i: ',i)
        #            print('no anda j: ',j)
        #MLR.perform_regression(pd.DataFrame(ts_dict).apply(stand,axis=0),pd.DataFrame(ts_dict).keys().insert(0,'clim'),config["work_dir"]+'/u850_1900_2099/'+alias)
        MLR.definitions(pd.DataFrame(ts_dict).keys().insert(0,'clim'))
        #Plot coefficients
        os.chdir(config["plot_dir"])
        os.getcwd()
        os.makedirs("u850_regressions_1900_2022",exist_ok=True)
        MLR.plot_regression_coef_map(config["work_dir"],'u850_1900_2099',alias,config["plot_dir"]+'/u850_regressions_1900_2099')
        MLR.plot_regression_lmg_map(config["work_dir"],'u850_1900_2099',alias,config["plot_dir"]+'/u850_regressions_1900_2099')
          
if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                                    
