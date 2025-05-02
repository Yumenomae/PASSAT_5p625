from .PASSAT import PASSAT

def build_model(config, adj_matrix, edgeMatrix):
    model_type = config.MODEL.TYPE

    if model_type == 'PASSAT':
        from data import grid2sphere, get_edge_node_aggregation, get_haversine_distance_from_sequence
        import torch
        lat_lon_mesh = config.DATA.LAT_LON_MESH[0]
        latlonMatrix = lat_lon_mesh.half().cuda().flatten(1).transpose(0, 1)
        edgeIndex, edge2node = get_edge_node_aggregation(edgeMatrix)

        edge2node = edge2node.to_sparse_coo()
        edge2node_Sum = torch.sparse.sum(edge2node, dim=[1]).to_dense()
        b = torch.diag(1 / edge2node_Sum)
        edge2node = torch.sparse.mm(b, edge2node).to_sparse_csr()

        edge2node = edge2node.half().cuda()
        edgeRadian = latlonMatrix[edgeIndex].transpose(0, 1)

        radianDiff = torch.abs(edgeRadian[0] - edgeRadian[1])
        radianDiff = torch.where(radianDiff <= torch.pi, radianDiff, 2*torch.pi - radianDiff)
        dist = get_haversine_distance_from_sequence(edgeRadian)

        edgeStates = torch.cat([radianDiff, dist], dim=1) # (N_e, 3)
        model = PASSAT(config, adj_matrix, edgeIndex, edge2node, edgeStates)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
