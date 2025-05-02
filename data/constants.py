import torch


def get_constants(
    mesh, lsm, oro
):
    lat = mesh[0] # (32, 64)
    lon = mesh[1]
    return torch.stack([lat, lon, lsm, oro])