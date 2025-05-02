import torch

class Criterion():

    def __init__(self, config):
        self.MSE = torch.nn.MSELoss()
        self.mesh = config.DATA.LAT_LON_MESH[0]
        self.lamda1 = config.MODEL.PASSAT.LAMBDA_VELOCITY_VALUE
        self.lamda2 = config.MODEL.PASSAT.LAMBDA_VELOCITY_GRAD

        weights_lat = torch.cos(self.mesh[0]) # (32, 64)
        weights_lat /= weights_lat.mean()   # (32, 64)
        weights_lat = weights_lat[None, None, None, :, :].cuda() # (1, 1, 1, 32, 64)

        self.weights_lat = weights_lat
    
    def latitude_weighted_MSE(self, predict, target):
        weights_error = ((predict-target) **2 * self.weights_lat).mean() # (T, B, 5, 32, 64)

        return weights_error

    def forward(self, predict_dataStates, targetStates, predict_velocity):
        dist = self.latitude_weighted_MSE(predict_dataStates, targetStates)

        zeros = torch.zeros_like(predict_velocity)
        lat_velocityPartial = torch.gradient(predict_velocity, dim=3, spacing=5.625)[0]
        lon_velocityPartial = torch.gradient(predict_velocity, dim=4, spacing=5.625)[0] 
        rectify_lon_velocityPartial = (lon_velocityPartial / torch.cos(self.mesh[0]).view(1, 1, 1, 32, 64))   
        distVel = self.MSE(predict_velocity, zeros)
        loss = dist + self.lamda1 * distVel + self.lamda2 * (self.MSE(lat_velocityPartial, zeros) + self.MSE(rectify_lon_velocityPartial, zeros))
        
        return loss, dist, distVel

class Validation():

    def __init__(self, config):
        mesh = config.DATA.LAT_LON_MESH[0]
        dataMean = config.DATA.DATAMEAN[0]
        dataStd = config.DATA.DATASTD[0]
        dataClim = config.DATA.DATACLIM[0]

        weights_lat = torch.cos(mesh[0]) # (32, 64)
        weights_lat /= weights_lat.mean()   # (32, 64)
        weights_lat = weights_lat[None, None, None, :, :].cuda() # (1, 1, 1, 32, 64)

        self.weights_lat = weights_lat
        self.dataMean = dataMean[None, None, :, None, None]
        self.dataStd = dataStd[None, None, :, None, None]
        self.dataClim = dataClim[None, None, :, :, :]

    def compute_weighted_rmse(self, dataStates, targetStates):

        dataStates = dataStates * self.dataStd + self.dataMean
        targetStates = targetStates * self.dataStd + self.dataMean

        weights_error = ((dataStates - targetStates) **2 * self.weights_lat).flatten(3) # (T, B, 5, 32 * 64)
        rmse = torch.sqrt(torch.mean(weights_error, dim=-1))   # (T, B, 5)

        return rmse

    def compute_weighted_acc(self, dataStates, targetStates):
 
        dataStates = dataStates * self.dataStd + self.dataMean # (T, B, 5, 32, 64)
        targetStates = targetStates * self.dataStd + self.dataMean # (T, B, 5, 32, 64)

        dataStates_ = dataStates - self.dataClim # (T, B, 5, 32, 64)
        targetStates_ = targetStates - self.dataClim # (T, B, 5, 32, 64)

        del dataStates 
        del targetStates

        mean_dataStates_ = dataStates_.flatten(3).mean(dim=3)   # (T, B, 5)
        mean_targetStates_ = targetStates_.flatten(3).mean(dim=3)   # (T, B, 5)

        dataStates_ = dataStates_ - mean_dataStates_[:, :, :, None, None]   # (T, B, 5, 32, 64)
        targetStates_ = targetStates_ - mean_targetStates_[:, :, :, None, None] # (T, B, 5, 32, 64)

        dataStates_ = dataStates_.flatten(3)
        targetStates_ = targetStates_.flatten(3)
        weights_lat = self.weights_lat.flatten(3)

        acc = (
                torch.sum(weights_lat * dataStates_ * targetStates_, dim=3) /
                torch.sqrt(
                    torch.sum(weights_lat * dataStates_ ** 2, dim=3) * torch.sum(weights_lat * targetStates_ ** 2, dim=3)
                )
        )   # (T, B, 5)

        return acc
            

    
