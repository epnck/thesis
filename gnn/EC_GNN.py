import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GraphNorm, global_mean_pool


class Mobility_ECGNN(nn.Module):
    def __init__(self, node_features_dim, edge_features_dim, hidden_dim):
        super(Mobility_ECGNN, self).__init__()

        # 1. Edge-Conditioning Network 1
        # Maps edge features (mobility volume) to a weight matrix for the nodes
        edge_net1 = nn.Sequential(
            nn.Linear(edge_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_features_dim * hidden_dim)
        )
        # NNConv is the standard Edge-Conditioned Convolution in PyG
        self.conv1 = NNConv(node_features_dim, hidden_dim, edge_net1, aggr='mean')

        self.norm1 = GraphNorm(hidden_dim)

        # 2. Edge-Conditioning Network 2
        edge_net2 = nn.Sequential(
            nn.Linear(edge_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_net2, aggr='mean')
        self.norm2 = GraphNorm(hidden_dim)

        # 3. Graph-Level Readout (Threshold Predictor)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)  # Outputs the predicted indicator or lambda_c

    def forward(self, data):
        # x: Node features (e.g., populations in S, E, I, R states)
        # edge_index: Graph connectivity (who travels to whom)
        # edge_attr: Mobility flow volume (e.g., # of commuters)
        # batch: Indicates which graph in the batch each node belongs to
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Pass 1: Nodes communicate, scaled entirely by the mobility edge network
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.norm1(x)


        # Pass 2: Deeper structural learning
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.norm2(x)


        # Global Pooling: Aggregate all city embeddings into one network-level embedding
        graph_embedding = global_mean_pool(x, batch)

        # Final Prediction Layer (Replacing the f_det from the TGNN paper)
        out = F.relu(self.fc1(graph_embedding))
        prediction = self.fc2(out)

        return prediction