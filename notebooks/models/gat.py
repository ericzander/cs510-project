"""
Attention based GCN implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(
            self, 
            in_channels,
            conv_channels: list,
            mlp_channels: list,
            out_channels,
            heads=4,
            concat=True,
            skip_connections=True,
            long_residual=True
        ):
        super(GAT, self).__init__()

        self.skip_connections = skip_connections
        self.long_residual = long_residual

        self.n_conv_layers = len(conv_channels)
        self.n_mlp_layers = len(mlp_channels)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)

        concat_heads = heads if concat else 1

        self.convs = nn.ModuleList([
            GATConv(in_channels, conv_channels[0], heads, concat=concat),
            *[GATConv(conv_channels[i - 1] * concat_heads, conv_channels[i], heads, concat=concat)
              for i in range(1, self.n_conv_layers)],
            GATConv(conv_channels[-1] * concat_heads, mlp_channels[0], heads, concat=concat),
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(conv_channels[0] * concat_heads),
            *[nn.LayerNorm(conv_channels[i] * concat_heads) for i in range(1, self.n_conv_layers)],
            nn.LayerNorm(mlp_channels[0] * concat_heads),
        ])

        self.projections = nn.ModuleList([
            nn.Linear(in_channels, conv_channels[0] * concat_heads),
            *[nn.Linear(conv_channels[i - 1] * concat_heads, conv_channels[i] * concat_heads)
              for i in range(1, self.n_conv_layers)],
            nn.Linear(conv_channels[-1] * concat_heads, mlp_channels[0] * concat_heads),
        ])

        self.input_projection = nn.Linear(in_channels, mlp_channels[0] * concat_heads)

        self.fc1 = nn.Linear(mlp_channels[0] * concat_heads, mlp_channels[1])
        self.fc2 = nn.Linear(mlp_channels[1], out_channels)
        
    def forward(self, x, edge_index, return_emb=False):
        embeddings = []

        start = x
        skip = x

        for i, conv in enumerate(self.convs):
            # Convolution
            x = conv(x, edge_index)
            if return_emb:
                embeddings.append(x)

            # Skip connections w/ linear projection
            if self.skip_connections:
                x = x + self.projections[i](skip)
                skip = x

            # Batch normalization
            x = self.layer_norms[i](x)

            # Activation + dropout
            x = self.gelu(x)
            x = self.dropout(x)

        # Residual
        if self.long_residual:
            start = self.input_projection(start)
            x = start + x

        # MLP
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        if return_emb:
            return output, embeddings

        return output
    