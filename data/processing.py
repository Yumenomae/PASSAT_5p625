import os
from tqdm import tqdm
import torch
from .loading import GetDataFrom_wb1
from .constants import get_constants

def make_constants(path):

    data = GetDataFrom_wb1(path, '2000')  # Loading leap year data

    lat = torch.tensor(data["lat"].values) * torch.pi / 180
    lon = torch.tensor(data["lon"].values) * torch.pi / 180
    mesh = torch.stack(torch.meshgrid(lat, lon))

    orography = data["orography"]
    orography = (orography - orography.mean())/orography.std()

    constants = get_constants(mesh, 
                              torch.tensor(data["lsm"].values),
                              torch.tensor(orography.values),
                              ).float()  # .to(device=device)  # if enough VRAM

    torch.save(constants, './Storages/constants')
    torch.save(mesh, './Storages/lat_lon_mesh')
    print('Successfully making constants!')
    return 
        


def make_dataList(path, periods):

    os.makedirs('./Storages/DataStorage', exist_ok=True)
    root = './Storages/DataStorage/'
    list = []
    with tqdm(total = len(periods)) as pbar:
        for year in periods:
            raw_data = GetDataFrom_wb1(path, year)
            for idx in range(len(raw_data['time']) // 6):
                print(raw_data['time'][idx*6].values)
                # print(f'Processing {raw_data['time'][idx*6]}')
                data = torch.stack([
                    torch.tensor(raw_data['t2m'][idx*6].values),
                    torch.tensor(raw_data['t'][idx*6].values),
                    torch.tensor(raw_data['z'][idx*6].values),
                    torch.tensor(raw_data['u10'][idx*6].values),
                    torch.tensor(raw_data['v10'][idx*6].values)
                ])
                name = root + str(year) + '_' + str(idx*6) + '_' + 'data'
                list.append(name)
                torch.save(data, name)
            pbar.update(1)

    torch.save(list, './Lists/dataList_1979_2018')
    print('Successfully making dataset!')
    return

def make_dataStat(path, year):

    os.makedirs('./Storages/DataStat', exist_ok=True)
    root = './Storages/DataStat/'
    raw_data = GetDataFrom_wb1(path, year)

    dataMean = torch.stack([
        torch.tensor(raw_data['t2m'].mean().values),
        torch.tensor(raw_data['t'].mean().values),
        torch.tensor(raw_data['z'].mean().values),
        torch.tensor(raw_data['u10'].mean().values),
        torch.tensor(raw_data['v10'].mean().values)
    ])
    dataStd = torch.stack([
        torch.tensor(raw_data['t2m'].std().values),
        torch.tensor(raw_data['t'].std().values),
        torch.tensor(raw_data['z'].std().values),
        torch.tensor(raw_data['u10'].std().values),
        torch.tensor(raw_data['v10'].std().values)
    ])
    dataClim = torch.stack([
        torch.tensor(raw_data['t2m'].mean('time').values),
        torch.tensor(raw_data['t'].mean('time').values),
        torch.tensor(raw_data['z'].mean('time').values),
        torch.tensor(raw_data['u10'].mean('time').values),
        torch.tensor(raw_data['v10'].mean('time').values)
    ])
    torch.save(dataMean, root + 'dataMean')
    torch.save(dataStd, root + 'dataStd')
    torch.save(dataClim, root + 'dataClim')
    print('Successfully making data statistics!')
    return

def seperate_dataList():
    dataList = torch.load('./Lists/dataList_1979_2018')

    train_dataList = dataList[:54056]
    valid_dataList = dataList[54056:55520]
    test_dataList = dataList[55520:]
    print('train_data start from ' + train_dataList[0] + ' to ' + train_dataList[-1])
    print('valid_data start from ' + valid_dataList[0] + ' to ' + valid_dataList[-1])
    print('test_data start from ' + test_dataList[0] + ' to ' + test_dataList[-1])
    
    torch.save(train_dataList, './Lists/dataList_1979_2018_train')
    torch.save(valid_dataList, './Lists/dataList_1979_2018_valid')
    torch.save(test_dataList, './Lists/dataList_1979_2018_test')
    return 



def grid2sphere(mesh):
    lat, lon = mesh[0, :, :], mesh[1, :, :]
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack((x, y, z), dim=0)

def grid2tanbun(mesh):
    lat, lon = mesh[0,:,:], mesh[1,:,:]

    # e_theta
    x = - torch.sin(lat) * torch.cos(lon)
    y = - torch.sin(lat) * torch.sin(lon)
    z = torch.cos(lat)
    e_theta = torch.stack((x,y,z), dim=0)

    # e_phi
    x = - torch.sin(lon)
    y = torch.cos(lon)   
    z = torch.zeros_like(x)
    e_phi = torch.stack((x,y,z), dim=0)

    return e_theta, e_phi

def sphere2tanbun(mesh):
    x, y, z = mesh[0,:,:], mesh[1,:,:], mesh[2,:,:]

    # e_theta
    x0 = - x * z
    y0 = - y * z
    z0 = x ** 2 + y ** 2
    e_theta = torch.stack((x0,y0,z0), dim=0)
    e_theta = e_theta / torch.norm(e_theta, 2, 0)
 
    # e_phi
    x0 = - y
    y0 = x
    z0 = torch.zeros_like(x0)
    e_phi = torch.stack((x0,y0,z0), dim = 0)
    e_phi = e_phi / torch.norm(e_phi, 2, 0)

    return e_theta, e_phi

def Wind2Vel(u, v, mesh):

    e_theta, e_phi = grid2tanbun(mesh)
    # suppose the radius of earth: 6357 km; 
    u, v = u * 3.6 / 6357, v * 3.6 / 6357 
    Vel = v * e_theta + u * e_phi

    return Vel

def get_dist(mesh, type=2):

    Coords_Cartesian= grid2sphere(mesh) 
    Coords_Cartesian_flatten = torch.flatten(Coords_Cartesian, 1)
    # Calculate pairwise distances
    Coords_Cartesian_diff = Coords_Cartesian_flatten[:,:,None] - Coords_Cartesian_flatten[:,None,:]
    dist_mat = torch.norm(Coords_Cartesian_diff, type, dim=0)
    return dist_mat

def get_haversine_distance(mesh):

    node_flatten = torch.flatten(mesh, 1)
    node_radius_diff = node_flatten[:, :, None] - node_flatten[:, None, :]
    first_node_radius = node_flatten[:, :, None].repeat(1, 1, node_flatten.size(1))
    second_node_radius = node_flatten[:, None, :].repeat(1, node_flatten.size(1), 1)
    a = torch.sin(node_radius_diff[0] /2 ) ** 2 + torch.cos(first_node_radius[0]) * torch.cos(second_node_radius[0]) * torch.sin(node_radius_diff[1] /2 ) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return c

def get_haversine_distance_from_sequence(s):
    # (2, N, 2)) to (N, 1)
    first_node, second_node = s[0], s[1] # (N, 2)
    latitude_diff = first_node[:, 0] - second_node[:, 0]
    longitude_diff = first_node[:, 1] - second_node[:, 1]
    a = torch.sin(latitude_diff/2) ** 2 + torch.cos(first_node[:, 0]) * torch.cos(second_node[:, 0]) * torch.sin(longitude_diff / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return c[:, None]

def get_adjacency(alpha, k):

    mesh = torch.load('./Storages/lat_lon_mesh')
    dist_mat = get_haversine_distance(mesh)
    sim_mat = torch.exp(- dist_mat ** 2 / (2 * alpha**2))
    zeros_mat = torch.zeros_like(sim_mat)
    threshold = torch.min(torch.topk(sim_mat, k=k, dim=1)[0])
    sim_mat = torch.where(sim_mat >= threshold, sim_mat, zeros_mat)
    D = torch.diag(torch.sum(sim_mat, dim=0) ** (-1/2)) # Renormalization matrix
    adj_mat = D @ sim_mat @ D # Renormalization
    edge_mat = adj_mat * torch.triu(torch.ones_like(adj_mat), diagonal=0)
    edge_mat = edge_mat - torch.diag(torch.diag(edge_mat))
    return adj_mat, edge_mat

def get_edge_node_aggregation(sparse_adjMatrix):

    sparse_adjMatrix = sparse_adjMatrix.to_sparse_coo()
    edgeIndex = sparse_adjMatrix.indices().transpose(0, 1)  # (N_e, 2)
    idx = []
    for edgeIdx in range(len(edgeIndex)):
        senderIdx = edgeIndex[edgeIdx][0]
        receiveIdx = edgeIndex[edgeIdx][1]
        idx.append(torch.tensor([senderIdx,  edgeIdx]))
        idx.append(torch.tensor([receiveIdx,  edgeIdx]))
    idx = torch.stack(idx)  # (2 * N_e, 2)
    rowIdx, colIdx = idx[:, 0], idx[:, 1]
    edge2node_matrix = torch.sparse_coo_tensor(indices=torch.stack([rowIdx, colIdx]), values=torch.ones(len(idx)))   # (N_n, N_e)
    return edgeIndex, edge2node_matrix.to_sparse_csr()