import torch
import numpy as np
import xarray as xr


def GetDataFrom_wb1(data_path, year):

    subfolders_list_path = [
        "/2m_temperature",
        "/temperature_850hPa",
        "/geopotential_500hPa",
        "/10m_u_component_of_wind",
        "/10m_v_component_of_wind",
    ]

    constant_data_path = "/constants/constants_5.625deg.nc"

    raw_data = []
    for subfolder_name in subfolders_list_path:
        dataset_path = data_path + subfolder_name + subfolder_name + f'_{year}_5.625deg.nc'
        data = xr.open_dataset(dataset_path)
        if "level" in data.coords:
            data = data.drop_vars("level")
        raw_data.append(data)

    merged_raw_data = xr.merge(raw_data)
    
    # Add constants to the data
    constants = xr.open_dataset(data_path + constant_data_path)


    constants = constants[['orography', 'lsm']]
    merged_raw_data = xr.merge([merged_raw_data, constants])

    return merged_raw_data

def get_constants(path):

    constants = xr.open_mfdataset(path, combine="by_coords")
    oro = torch.tensor(constants["orography"].values)[(None,) * 2]
    lsm = torch.tensor(constants["lsm"].values)[(None,) * 2]
    lat2d = torch.tensor(constants["lat2d"].values)
    lon2d = torch.tensor(constants["lon2d"].values)
    return (
        oro,
        lsm,
        lat2d,
        lon2d,
    ) 