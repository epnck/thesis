import torch.nn as nn
from torch_geometric.nn import GATConv,  LayerNorm, global_mean_pool
import torch.nn.functional as F

class T_GNN (nn.Module):

    def __init__(self, node_dimension, hidden_dimension):
        super(T_GNN, self).__init__()

        self.linear1 = nn.Linear(node_dimension, hidden_dimension)

        #ecc, output is of dimension hidden dimensits
        self.GATconv1 = GATConv(hidden_dimension, hidden_dimension)

        #graph norm pass 1
        self.layernorm1 = LayerNorm(hidden_dimension)

        #fully connected layer
        self.fcl1 = nn.Linear(hidden_dimension, 1)


    def forward(self, data):

        node_features = data.x #popullation, area , position
        edge_index =  data.edge_index
        # flow = data.edge_attr #mobility flow

        batch = data.batch
        # f_in from T-GNN
        x = self.linear1(node_features)
        x = F.relu(x)

        #f_att from tgnn
        x = self.GATconv1(x, edge_index)
        x = F.relu(x)
        #f_norm from tgnn
        x = self.layernorm1(x)

        #f_out
        embedding = global_mean_pool(x, batch)

        #adpated prediciton head
        f_out = self.fcl1(embedding)
        predicion = F.relu(f_out)

        return predicion
