import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GraphNorm
from torch_geometric.nn.aggr import AttentionalAggregation


class Advanced_Mobility_GNN(nn.Module):
    def __init__(self, edge_dimension, node_dimension, hidden_dimension):
        super(Advanced_Mobility_GNN, self).__init__()

        # --- PASS 1 ---
        edge_weights1 = nn.Sequential(
            nn.Linear(edge_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, node_dimension * hidden_dimension)
        )
        self.ECC1 = NNConv(node_dimension, hidden_dimension, edge_weights1, aggr='mean')
        self.graphnorm1 = GraphNorm(hidden_dimension)

        # --- PASS 2 ---
        edge_weights2 = nn.Sequential(
            nn.Linear(edge_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension * hidden_dimension)
        )
        self.ECC2 = NNConv(hidden_dimension, hidden_dimension, edge_weights2, aggr='mean')
        self.graphnorm2 = GraphNorm(hidden_dimension)

        # ==========================================
        # INNOVATION 1: Global Attention Pooling
        # A mini neural network that learns which zip codes matter most!
        # ==========================================
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.ReLU(),
            nn.Linear(hidden_dimension // 2, 1)  # Outputs a single importance score per node
        )
        self.attn_pool = AttentionalAggregation(gate_nn=self.attention_gate)

        # ==========================================
        # INNOVATION 2: Jumping Knowledge Dense Layers
        # Because we will concatenate the embeddings from Pass 1 AND Pass 2,
        # the input to this layer is now hidden_dimension * 2
        # ==========================================
        self.fcl1 = nn.Linear(hidden_dimension * 2, hidden_dimension)
        self.fcl2 = nn.Linear(hidden_dimension, 1)

    def forward(self, data):
        node_features = data.x
        edge_index = data.edge_index
        flow = data.edge_attr
        batch = data.batch

        # --- PASS 1 ---
        x1 = self.ECC1(node_features, edge_index, flow)
        x1 = F.relu(x1)
        x1 = self.graphnorm1(x1)

        # Create an embedding just for Pass 1's localized view
        embedding1 = self.attn_pool(x1, index=batch)

        # --- PASS 2 ---
        x2 = self.ECC2(x1, edge_index, flow)
        x2 = F.relu(x2)
        x2 = self.graphnorm2(x2)

        # INNOVATION 3: Residual / Skip Connection
        # We add the features from Pass 1 directly into the output of Pass 2
        x2 = x2 + x1

        # Create an embedding for Pass 2's broader view
        embedding2 = self.attn_pool(x2, index = batch)

        # --- COMBINE & PREDICT (Jumping Knowledge) ---
        # Glue the 1-hop view and 2-hop view together side-by-side
        final_embedding = torch.cat([embedding1, embedding2], dim=1)

        f_out = self.fcl1(final_embedding)
        f_out = F.relu(f_out)

        # Adding Dropout to the dense layer (great for preventing overfitting on advanced models!)
        f_out = F.dropout(f_out, p=0.3, training=self.training)

        prediction = self.fcl2(f_out)

        return prediction