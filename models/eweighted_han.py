import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from class_resolver.contrib.torch import activation_resolver
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import FAConv, HeteroConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.nn import HeteroConv, Linear, HANConv
from .eweightedhan_conv import EWeightedHANConv

# from torch_timeseries.layers.graphsage import MyGraphSage


class EWeightedHAN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        edge_dim,
        n_layers,
        dropout=0.0,
        heads=1,
        negative_slope=0.2,
        act="relu",
        n_first=True,
        act_first=False,
        conv_type='all', # homo, hetero
        # edge_dim=1,
        # edge_mode=0,
        **kwargs
    ):
        self.n_first = n_first

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = n_layers
        self.edge_dim = edge_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.conv_type = conv_type
        self.dropout = dropout
        self.act = activation_resolver.make(act)
        self.act_first = act_first

        self.out_channels = hidden_channels

        # assert (
        #     n_layers >= 2
        # ), "intra and inter conv layers must greater than or equals to 2 "

        
        # 定义元路径
        self.metadata = [
            ['f', 'o'],
            [('f', 'f2f', 'f'), ('o', 'o2o', 'o'), ('f', 'f2o', 'o'),('o', 'o2f', 'f')]
        ]

        
        self.convs = nn.ModuleList()
        
        if n_layers > 1:
            self.convs.append(self.init_conv(self.in_channels,hidden_channels))
        
        for i in range(n_layers - 2):
            self.convs.append(self.init_conv(self.hidden_channels,hidden_channels))
        if n_layers==1:
            self.convs.append(self.init_conv(self.in_channels,out_channels))
        else:
            self.convs.append(self.init_conv(self.in_channels,hidden_channels))
            
        # self.norms = None
        # if norm is not None:
        #     self.norms = nn.ModuleList()
        #     for _ in range(n_layers - 1):
        #         self.norms.append(copy.deepcopy(norm))


    def init_conv(self, in_channels, out_channels, **kwargs):
        han = EWeightedHANConv(in_channels, out_channels,self.edge_dim,self.metadata,heads=self.heads, negative_slope=self.negative_slope,dropout=self.dropout)
        return han
    
    def forward(self, x, edge_attr,edge_index, num_observations):
        # x: B * (N+T) * C
        # edge_index: B,2,2*(N*T)
        # edge_attr: B*E or B * (N * T )
        self.num_observations = num_observations

        for i in range(self.num_layers):
            x_dict = {
                "o": x[: self.num_observations, :],
                "f": x[self.num_observations :, :],
            }
            edge_index_bi = edge_index
            edge_weight_bi = edge_attr
            # TODO: edge may be empty, please ensure no empty edges here
            # assert ((edge_index_bi[0] < self.num_observations) 
            #     & (edge_index_bi[1] < self.num_observations)).any() == True
            
            oo_index = (edge_index_bi[0] < self.num_observations)  & (edge_index_bi[1] < self.num_observations)
            ff_index = (edge_index_bi[0] >= self.num_observations) & (edge_index_bi[1] >= self.num_observations)
            of_index = (edge_index_bi[0] < self.num_observations)  & (edge_index_bi[1] >= self.num_observations)
            fo_index = (edge_index_bi[0] >= self.num_observations) & (edge_index_bi[1] < self.num_observations)
            
            edge_ff = edge_index_bi[:,ff_index,]
            edge_ff_weight = edge_weight_bi[ff_index,]
            
            edge_oo = edge_index_bi[:,oo_index,] 
            edge_oo_weight = edge_weight_bi[oo_index,]
            
            edge_fo = edge_index_bi[:,fo_index,]
            edge_fo_weight = edge_weight_bi[fo_index,]

            edge_of = edge_index_bi[:,of_index,]
            edge_of_weight = edge_weight_bi[of_index,]

            # convert edge index to edge index dict
                
            edge_ff = edge_ff - self.num_observations 
            edge_of[1, :] = edge_of[1, :] - self.num_observations
            edge_fo[0, :] = edge_fo[0, :] - self.num_observations
            
            
            edge_index_dict = {}
            if self.conv_type == 'all':
                edge_index_dict = {
                    ("f", "f2f", "f"): edge_ff,
                    ("o", "o2o", "o"): edge_oo,
                    ("f", "f2o", "o"): edge_fo,
                    ("o", "o2f", "f"): edge_of,
                }
                
                edge_weight_dict = {
                    ("f", "f2f", "f"): edge_ff_weight,
                    ("o", "o2o", "o"): edge_oo_weight,
                    ("f", "f2o", "o"): edge_fo_weight,
                    ("o", "o2f", "f"): edge_of_weight,
                }
            elif self.conv_type == 'homo':
                
                edge_index_dict = {
                    ("f", "f2f", "f"): edge_ff,
                    ("o", "o2o", "o"): edge_oo,
                }
                edge_weight_dict = {
                    ("f", "f2o", "o"): edge_fo_weight,
                    ("o", "o2f", "f"): edge_of_weight,
                }

            elif self.conv_type == 'hetero':
                edge_index_dict = {
                    ("f", "f2o", "o"): edge_fo,
                    ("o", "o2f", "f"): edge_of,
                }
                edge_weight_dict = {
                    ("f", "f2o", "o"): edge_fo_weight,
                    ("o", "o2f", "f"): edge_of_weight,
                }
            else:
                raise NotImplementedError("conv_type must be 'all', 'homo' or 'heter'")
            out_dict,out_att = self.convs[i](x_dict, edge_index_dict,edge_weight_dict)
            x = torch.concat([out_dict["o"], out_dict["f"]], dim=0)
            # xs.append(xi)
                
            # x = torch.stack(xs)
            if i == self.num_layers - 1:
                break

            # if self.act_first:
            #     x = self.act(x)
            # # if self.norms is not None:
            #     x = self.norms[i](x)
            # if not self.act_first:
            #     x = self.act(x)

            # x = F.dropout(x, p=self.dropout, training=self.training)
        return x

