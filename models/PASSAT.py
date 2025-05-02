import math
import torch
import torch.nn as nn
from updates import build_update_method
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EmbeddingBlock(nn.Module):
    def __init__(self, embed_dim, drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.mlp_node = Mlp(in_features=9, hidden_features=embed_dim, out_features=embed_dim, act_layer=act_layer, drop=drop)
        self.mlp_edge = Mlp(in_features=3, hidden_features=embed_dim, out_features=embed_dim, act_layer=act_layer, drop=drop)


    def forward(self, node, edge):  # (B, N_e, D) / (B, N_n, D)

        node = self.mlp_node(node)
        edge = self.mlp_edge(edge)

        return node, edge

class GraphConnectionBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.norm1 = norm_layer(3 * in_features)
        self.norm2 = norm_layer(2 * in_features)
        self.mlpEdge = Mlp(in_features=3 * in_features, hidden_features=3 * hidden_features, out_features=out_features, drop=drop_rate)
        self.mlpNode = Mlp(in_features=2 * in_features, hidden_features=2 * hidden_features, out_features=out_features, drop=drop_rate)


    def forward(self, node, edge, edgeIdx, edge2node):  # (B, N_e, D) / (B, N_n, D)
        B = node.size(0)
        N_n = node.size(1)
        N_e = edge.size(1)
        shortcut_edge = edge
        shortcut_node = node

        edge = shortcut_node[:, edgeIdx].flatten(2)   # (B, N_e, 2, D) → (B, N_e, 2D)
        edge = torch.cat([shortcut_edge, edge], dim=2)  # (B, N_e, 3D)
        # del node2edge

        edge = self.norm1(edge)
        edge = self.mlpEdge(edge).transpose(0, 1).flatten(1).half() # (B, N_e, D) → (N_e, B*D)

        node = torch.sparse.mm(edge2node, edge).reshape(N_n, B, -1).transpose(0, 1)   # (N_n, B*D) → (N_n, B, D) → (B, N_n, D)
        node = torch.cat([shortcut_node, node], dim=2)  # (B, N_n, 2D)
        edge = edge.reshape(N_e, B, -1).transpose(0, 1)
        # del edge2node

        node = self.norm2(node)
        node = self.mlpNode(node)

        edge = shortcut_edge + edge
        node = shortcut_node + node

        return node, edge
    
class GraphConvolutionBlock(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, adj=None, norm_layer=nn.LayerNorm, drop_rate=0.):
        super(GraphConvolutionBlock, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.norm1 = norm_layer(in_features)
        self.norm2 = norm_layer(2 * in_features)
        self.Mlp1 = Mlp(in_features, hidden_features, in_features, drop=drop_rate)
        self.Mlp2 = Mlp(2 * in_features, hidden_features, out_features, drop=drop_rate)

    def forward(self, node, edge, adj):
        B = node.size(0)
        N_n = node.size(1)

        shortcut = node    # (B, N, in_chans)
        node = self.norm1(node)
        node = self.Mlp1(node) + shortcut

        shortcut = node

        node = node.transpose(0, 1).flatten(1).half()   # (B, N, hidden_chans) → (N, B*in_features)
        node = torch.sparse.mm(adj, node).reshape(N_n, B,-1).transpose(0, 1) # (B, N, in_features)

        node = torch.cat([shortcut, node], dim=2).type_as(shortcut)
        node = self.norm2(node)
        node = self.Mlp2(node) + shortcut
        return node, edge

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class BasicBlock(nn.Module):

    def __init__(self, dim, mlp_ratio, drop_rate):
        super(BasicBlock, self).__init__()
        self.GraphConnection = GraphConnectionBlock(in_features=dim, hidden_features=int(mlp_ratio * dim), out_features=dim, drop_rate=drop_rate)
        self.GraphConvolution = GraphConvolutionBlock(in_features=dim, hidden_features=int(mlp_ratio * dim), out_features=dim, drop_rate=drop_rate)

    def forward(self, node, edge, edgeIdx, edge2node, adj):
        node, edge = self.GraphConnection(node, edge, edgeIdx, edge2node)
        node, edge = self.GraphConvolution(node, edge, adj)
        return node, edge


class BasicLayer(nn.Module):

    def __init__(self, dim, depth, drop_rate=0.0, 
                 mlp_ratio= 4., dim_up=None, dim_down=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([BasicBlock(dim=dim, mlp_ratio=mlp_ratio, drop_rate=drop_rate)
            for i in range(depth)])

        if dim_up:
            self.processNode = Mlp(in_features=dim, hidden_features=dim, out_features=2 * dim, drop=drop_rate)
            self.processEdge = Mlp(in_features=dim, hidden_features=dim, out_features=2 * dim, drop=drop_rate)
        elif dim_down:
            self.processNode = Mlp(in_features=dim, hidden_features=dim, out_features=int(1 / 2 * dim), drop=drop_rate)
            self.processEdge = Mlp(in_features=dim, hidden_features=dim, out_features=int(1 / 2 * dim), drop=drop_rate)
        else:
            self.processNode = None

    def forward(self, node, edge, edgeIdx, edge2node, adj):
        for blk in self.blocks:
            node, edge = blk(node, edge, edgeIdx, edge2node, adj)
        if self.processNode:
            node = self.processNode(node)
            edge = self.processEdge(edge)
        return node, edge

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    
class Model(nn.Module):

    def __init__(self, edgeIdx, edge2node, edgeStates, embed_dim=96, 
                 adj=None, backbone_depths=[2, 2, 6, 2], 
                 branch_depths=[[2, 2], [2, 2]],
                 drop_rate=0., norm_layer=nn.LayerNorm):
        super(Model, self).__init__()

        self.edgeIdx = edgeIdx
        self.edge2node = edge2node
        self.edgeStates = edgeStates
        self.adj = adj
        self.embed= EmbeddingBlock(embed_dim)

        self.backbone_num_layers = len(backbone_depths)
        self.branch_num_layers = [len(branch_depths[i]) for i in range(len(branch_depths))]
        self.backbone_num_features = int(embed_dim * 2 ** (self.backbone_num_layers - 1))
        self.branch_num_features = [int(self.backbone_num_features * 2 ** -(self.branch_num_layers[i] - 1)) for i in range(len(branch_depths))]

        # build backbone_layers
        self.Backbone = nn.ModuleList()
        for i_layer in range(self.backbone_num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=backbone_depths[i_layer], 
                               drop_rate=drop_rate, mlp_ratio=4.0,
                               dim_up = True if (i_layer < self.backbone_num_layers - 1) else None)
            self.Backbone.append(layer)

        self.VelocityBranch = nn.ModuleList()
        for i_layer in range(self.branch_num_layers[0]):
            layer = BasicLayer(dim=int(self.backbone_num_features * 2 ** -i_layer), depth=branch_depths[0][i_layer],
                               drop_rate=drop_rate, mlp_ratio=4.0,
                               dim_down = True if (i_layer < self.branch_num_layers[0] - 1) else None)
            self.VelocityBranch.append(layer)
        self.velocity_norm = norm_layer(self.branch_num_features[0])
        self.velocity_head = Mlp(in_features=self.branch_num_features[0], hidden_features=self.branch_num_features[0], out_features=10)

        self.InteractionBranch = nn.ModuleList()
        for i_layer in range(self.branch_num_layers[1]):
            layer = BasicLayer(dim=int(self.backbone_num_features * 2 ** -i_layer), depth=branch_depths[1][i_layer],
                               drop_rate=drop_rate, mlp_ratio=4.0,
                               dim_down = True if (i_layer < self.branch_num_layers[1] - 1) else None)
            self.InteractionBranch.append(layer)
        self.interaction_norm = norm_layer(self.branch_num_features[1])
        self.interaction_head = Mlp(in_features=self.branch_num_features[1], hidden_features=self.branch_num_features[1], out_features=5)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def inputGeneration(self, dataStates, constants):
        B = dataStates.size(0)
        edge = self.edgeStates[None, :, :].repeat(B, 1, 1).type_as(dataStates) 
        node = torch.cat([dataStates, constants], dim=1).flatten(2).transpose(1, 2)  
        return node, edge
    
    def backbone_forward_features(self, node, edge): 
        node, edge = self.embed(node, edge)

        for layer in self.Backbone:
            node, edge = layer(node, edge, self.edgeIdx, self.edge2node, self.adj)

        return node, edge

    def VelocityBranch_forward_features(self, node, edge):
        for layer in self.VelocityBranch:
            node, edge = layer(node, edge, self.edgeIdx, self.edge2node, self.adj)

        node = self.velocity_norm(node)

        return node
    
    def InteractionBranch_forward_features(self, node, edge):
        for layer in self.InteractionBranch:
            node, edge = layer(node, edge, self.edgeIdx, self.edge2node, self.adj)

        node = self.interaction_norm(node)

        return node

    def forward(self, dataStates, constants):
        node, edge = self.inputGeneration(dataStates, constants)
        node, edge = self.backbone_forward_features(node, edge)

        node_iv = self.VelocityBranch_forward_features(node, edge)
        node_pp = self.InteractionBranch_forward_features(node, edge)

        del node
        del edge

        node_iv = self.velocity_head(node_iv)
        node_pp = self.interaction_head(node_pp)

        return node_iv, node_pp

class PASSAT(nn.Module):

    def __init__(self, config, adj, edgeIdx, edge2node, edgeStates):
        super(PASSAT, self).__init__()

        self.model = Model(edgeIdx=edgeIdx, edge2node=edge2node, edgeStates=edgeStates,
                                 adj=adj, embed_dim=config.MODEL.PASSAT.EMBED_DIM,
                                 backbone_depths=config.MODEL.PASSAT.BACKBONE_DEPTHS,
                                 branch_depths=config.MODEL.PASSAT.BRANCH_DEPTHS)
        self.updateMethod = build_update_method(config)
        self.mesh = config.DATA.LAT_LON_MESH[0]
        self.constants = config.DATA.CONSTANTS[0]

    def forward(self, dataStates, step):
        updatedData, updatedVelocity = self.updateMethod(self.mesh, self.constants, dataStates, self.model, step)  # (T, B, 5, 32, 64)
        return updatedData, updatedVelocity


