import torch

def AdvectionCore(lat_velocity, lon_velocity, lat_dataPartial, lon_dataPartial):
    ConvectiveDerivatives = lat_velocity * lat_dataPartial + lon_velocity * lon_dataPartial
    return - ConvectiveDerivatives

def UpdateVariables(dataStates, velocityCoef, 
                    lat_dataPartial,
                    rectify_lon_dataPartial,
                    InteractionTendencies):
    lat_velocityCoef, lon_velocityCoef = velocityCoef[:, :5], velocityCoef[:, 5:] # (B, 5, 32, 64)
    InteractionTendencies = InteractionTendencies.transpose(1, 2).view(-1, 5, 32, 64) # (B, 5, 32, 64)
    AdvectionTendencies = AdvectionCore(lat_velocityCoef, lon_velocityCoef, lat_dataPartial, rectify_lon_dataPartial) # (B, 5, 32, 64)
    localPartial = AdvectionTendencies + InteractionTendencies  # (B, 5, 32, 64)
    updated_dataStates = dataStates + localPartial * (1/5)  # 1 stands for the time 1 hour. Changes equal to velocity * time (Euler)
    return updated_dataStates.float()    

def UpdateVelocity(velocityCoef, mesh, lat_presPartial, rectify_lon_presPartial):
    lat_presPartial = lat_presPartial[:, None, :, :].repeat(1, 5, 1, 1)
    rectify_lon_presPartial = rectify_lon_presPartial[:, None, :, :].repeat(1, 5, 1, 1)

    latVelocity_pressureGradient = 1e-3 * lat_presPartial # Change the unit into (6731km)^2 * hour^−2; since we have normalize the geopoential, it just need to multiply 1e-3
    lonVelocity_pressureGradient = 1e-3 * rectify_lon_presPartial
    # Calculate the partial of velocity: lat_... means the gradient of latitude of all velocity component
    lat_velocityPartial = torch.gradient(velocityCoef, dim=2, spacing = 5.625)[0] # (B, 10, 32 ,64)
    # the gradient of lontitude of all velocity component
    lon_velocityPartial = torch.gradient(velocityCoef, dim=3, spacing = 5.625)[0] # (B, 10, 32 ,64)
    rectify_lon_velocityPartial = (lon_velocityPartial / torch.cos(mesh[0]).view(1, 1, 32, 64)) # (B, 10, 32, 64)

    latVelocity_Partial2lat =  lat_velocityPartial[:, :5]   # (B, 5, 32, 64)
    latVelocity_Partial2lon = rectify_lon_velocityPartial[:, :5]    # (B, 5, 32, 64)
    lonVelocity_Partial2lat = lat_velocityPartial[:, 5:]    # (B, 5, 32, 64)
    lonVelocity_Partial2lon = rectify_lon_velocityPartial[:, 5:]    # (B, 5, 32, 64)

    lat_velocityCoef, lon_velocityCoef = velocityCoef[:, :5], velocityCoef[:, 5:] # (B, 5, 32, 64)

    # Calculate the convective partial.
    latVelocity_convectivePartial = lat_velocityCoef * latVelocity_Partial2lat + \
                                    lon_velocityCoef * latVelocity_Partial2lon # (B, 5, 32, 64)
    lonVelocity_convectivePartial = lat_velocityCoef * lonVelocity_Partial2lat + \
                                    lon_velocityCoef * lonVelocity_Partial2lon # (B, 5, 32, 64) 
    
    tan_latitude = torch.tan(mesh[0]).view(1, 1, 32, 64)
    sin_latitude = torch.sin(mesh[0]).view(1, 1, 32, 64)
    cos_latitude = torch.cos(mesh[0]).view(1, 1, 32, 64)

    latVelocity_curvature = lon_velocityCoef ** 2 * tan_latitude
    lonVelocity_curvature = - lon_velocityCoef * lat_velocityCoef * tan_latitude

    latVelocity_Coriolis = - 2 * 0.2618 * lon_velocityCoef * sin_latitude
    lonVelocity_Coriolis = 2 * 0.2618 * lat_velocityCoef * sin_latitude
    
    latVelocity_Laplace = 1e-4 * lat_velocityCoef / (cos_latitude ** 2)
    lonVelocity_Laplace = 1e-4 * lon_velocityCoef / (cos_latitude ** 2)

    latVelocity_localPartial = - latVelocity_pressureGradient \
                            - latVelocity_convectivePartial \
                            - latVelocity_curvature \
                            + latVelocity_Coriolis \
                            - latVelocity_Laplace \

    lonVelocity_localPartial = - lonVelocity_pressureGradient \
                            - lonVelocity_convectivePartial \
                            - lonVelocity_curvature \
                            + lonVelocity_Coriolis \
                            - lonVelocity_Laplace \
            
    lat_velocityCoef = lat_velocityCoef + latVelocity_localPartial * (1/5)    # (B, 5, 32, 64): 1 stands for the time 1 hour. Changes equal to velocity * time
    lon_velocityCoef = lon_velocityCoef + lonVelocity_localPartial * (1/5)   # (B, 5, 32, 64)
    
    return torch.cat([lat_velocityCoef, lon_velocityCoef], dim=1).float() # (B, 10, 32, 64)

def update(mesh, constants, dataStates, model, step):

    updatedData, updatedVelocity = [], []
    velocityCoef = None
    B = dataStates.size(0)
    constants = constants[None, :, :, :].repeat(B, 1, 1, 1)

    for t in range(step):
        for t_phy in range(6):
            MotionFields, InteractionTendencies = model(dataStates, constants) # (B, 32*64, 10), (B, 32*64, 5)
            if velocityCoef == None:
                velocityCoef = MotionFields.transpose(1,2).view(-1, 10, 32, 64)
                velocityCoef = torch.where(torch.abs(velocityCoef)<=0.005, velocityCoef, torch.sign(velocityCoef) * 0.005) # Pre-Processing
                updatedVelocity.append(MotionFields.transpose(1,2).view(-1, 10, 32, 64))

            for t_states in range(5):
                lat_dataPartial = torch.gradient(dataStates, dim=2, spacing=5.625)[0] # (B, 5, 32 ,64)  # The interval of latitude and lontitude is 5.625°
                lon_dataPartial = torch.gradient(dataStates, dim=3, spacing=5.625)[0] # (B, 5, 32 ,64)
                rectify_lon_dataPartial = (lon_dataPartial / torch.cos(mesh[0]).view(1, 1, 32, 64)) # (B, 5, 32 ,64) #         
                # Gradient in Cartesian coordiantes = (1/r)f_{lat}e_{lat}+(1/rcos(lat))f_{lon}e_{phi}

                dataStates = UpdateVariables(dataStates, velocityCoef, lat_dataPartial, rectify_lon_dataPartial, InteractionTendencies)
                velocityCoef = UpdateVelocity(velocityCoef, mesh, lat_dataPartial[:, 2], rectify_lon_dataPartial[:, 2])
                # velocityCoef = torch.where(torch.abs(velocityCoef)<=0.005, velocityCoef, torch.sign(velocityCoef) * 0.005) # Post-Processing

        updatedData.append(dataStates)

    return torch.stack(updatedData), torch.stack(updatedVelocity)