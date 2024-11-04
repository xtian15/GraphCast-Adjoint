import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from graphcast import checkpoint
from graphcast import graphcast
from gc_subs import loadInputs
from gc_subs import gcForecast

floatType=float

if __name__=='__main__':

    with open('../params/GraphCast_params_1.00deg_13levels.npz', 'rb') as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    params = ckpt.params
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    # float32 to float64
    for key in params.keys():
        if 'norm' in key:
            params[key]['offset']=params[key]['offset'].astype(floatType)
            params[key]['scale']=params[key]['scale'].astype(floatType)
        else:
            params[key]['b']=params[key]['b'].astype(floatType)
            params[key]['w']=params[key]['w'].astype(floatType)

    inputs, targets, forcings=loadInputs(
        task_config,
        fName='../dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc'
    )

    for i in inputs:
        inputs[i]=inputs[i].astype(floatType)
    for i in forcings:
        forcings[i]=forcings[i].astype(floatType)
    for i in targets:
        targets[i]=targets[i].astype(floatType)

    with open("../stats/diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xr.load_dataset(f).compute()
    with open("../stats/mean_by_level.nc", "rb") as f:
        mean_by_level = xr.load_dataset(f).compute()
    with open("../stats/stddev_by_level.nc", "rb") as f:
        stddev_by_level = xr.load_dataset(f).compute()

    fwd=gcForecast(
        params, diffs_stddev_by_level, mean_by_level, stddev_by_level,
        staticFile='./Static_Res_1.00deg.npz',
    )

    predictions_ad=targets*0.
    holder=predictions_ad['temperature'].sel(
        #level=250, lat=slice(30,36), lon=slice(137,143))
        level=250, lat=33, lon=140)
    holder[()]=1.

    inputs_ad=fwd.forecast_adj(predictions_ad, inputs, forcings, targets)
    #inputs_ad.to_netcdf(path='InputsAD-T250-7x7Points.nc', mode='w')
    inputs_ad.to_netcdf(path='InputsAD-T250-1Point.nc', mode='w')
    print('done with adjoint run')
