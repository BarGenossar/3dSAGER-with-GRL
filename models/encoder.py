import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn import global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, feature_config, model_params):
        """
        GraphSAGE-based encoder for heterogeneous graphs.

        Args:
            feature_config (dict): contains info about node/edge types.
            model_params (dict): contains the following training hyperparameters:
                hidden_dim1 (int): hidden dimension size.
                hidden_dim2 (int): hidden dimension size.
                out_dim (int): output embedding dimension.
                gnn_layers_num (int): number of GNN layers (1 to 3).
                dropout_rate (float): dropout rate.
        """
        super().__init__()
        self.hidden_dim1 = model_params['hidden_dim1']
        self.hidden_dim2 = model_params['hidden_dim2']
        self.out_dim = model_params['out_dim']
        self.num_layers = model_params['gnn_layers_num']
        self.dropout = nn.Dropout(model_params['dropout_rate'])
        # verify that the number of gnn layers is at least 1 and at most 3, otherwise raise an error
        if self.num_layers < 1 or self.num_layers > 3:
            raise ValueError("Number of GNN layers must be between 2 and 3.")

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._define_layers(feature_config)
        self.project = nn.Linear(self.out_dim, self.out_dim)

    def _define_layers(self, feature_config):
        for i in range(self.num_layers):
            out_channels = self._get_out_channels(i)
            conv = HeteroConv(
                {
                    (src, rel, dst): SAGEConv((-1, -1), out_channels, aggr="mean")
                    for (src, rel, dst) in feature_config.get("relations", [])
                }, aggr="mean",
            )
            self.layers.append(conv)
            self.norms.append(nn.BatchNorm1d(out_channels))

    def _get_out_channels(self, layer_idx):
        if layer_idx == 0:
            return self.hidden_dim1
        elif layer_idx == 1:
            if self.num_layers == 2:
                return self.out_dim
            else:
                return self.hidden_dim2
        else:
            return self.out_dim

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for i, conv in enumerate(self.layers):
            x_dict = conv(x_dict, edge_index_dict)
            new_x = {}
            for node_type, x in x_dict.items():
                x = torch.relu(x)
                x = self.norms[i](x)
                x = self.dropout(x)
                new_x[node_type] = x
            x_dict = new_x
        if "MainObject" not in x_dict:
            raise ValueError("Expected 'MainObject' node type in graph.")
        batch = data["MainObject"].batch
        emb = global_mean_pool(x_dict["MainObject"], batch)
        return self.project(emb)
