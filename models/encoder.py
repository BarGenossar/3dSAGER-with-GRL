import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn import global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, feature_config, hidden_dim=64, out_dim=128, num_layers=2, dropout=0.2):
        """
        GraphSAGE-based encoder for heterogeneous graphs.

        Args:
            feature_config (dict): contains info about node/edge types.
            hidden_dim (int): hidden dimension size.
            out_dim (int): output embedding dimension.
            num_layers (int): number of GraphSAGE layers.
            dropout (float): dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            conv = HeteroConv(
                {
                    (src, rel, dst): SAGEConv(
                        (-1, -1),
                        hidden_dim if i < num_layers - 1 else out_dim,
                        aggr="mean",
                    )
                    for (src, rel, dst) in feature_config.get("relations", [])
                },
                aggr="mean",
            )
            self.layers.append(conv)
            if i < num_layers - 1:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norms.append(nn.BatchNorm1d(out_dim))
        self.project = nn.Linear(out_dim, out_dim)

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
