import torch.nn as nn
from torch_geometric.nn import GATConv,  GraphNorm, global_mean_pool
import torch.nn.functional as F

class GAT_GNN (nn.Module):

    def __init__(self, node_dimension, hidden_dimension):
        super(GAT_GNN, self).__init__()



        #ecc, output is of dimension hidden dimensits
        self.conv1 = GATConv(node_dimension, hidden_dimension,) #aggr mean to prevnt hubs from dominating

        #graph norm pass 1
        self.graphnorm1 = GraphNorm(hidden_dimension)


        #ecc
        self.conv2 = GATConv(hidden_dimension, hidden_dimension) #in channels is output from pass 1, so hidden dim
        # graph norm pass 2
        self.graphnorm2 = GraphNorm(hidden_dimension)


        # pass 3
        # edge_weights3 = nn.Sequential(
        #     nn.Linear(edge_dimension, hidden_dimension),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dimension, hidden_dimension * hidden_dimension)
        #
        # )
        #
        # self.ECC3 = NNConv(hidden_dimension, hidden_dimension, edge_weights3,
        #                    aggr='mean')  # in channels is output from pass 2, so hidden dim
        # # graph norm pass 3
        # self.graphnorm3 = GraphNorm(hidden_dimension)


        #fully connected layer
        self.fcl1 = nn.Linear(hidden_dimension, hidden_dimension //2)
        self.fcl2 =nn.Linear(hidden_dimension // 2 ,1)


    def forward(self, data):

        node_features = data.x #popullation, area , position
        edge_index =  data.edge_index
        # flow = data.edge_attr #mobility flow

        batch = data.batch

         #pass 1
        x = self.conv1(node_features, edge_index)
        x = self.graphnorm1(x)
        x = F.relu(x)

        #pass 2
        x = self.conv2(x, edge_index)
        x = self.graphnorm2(x)
        x = F.relu(x)


        # # pass 3
        # x = self.ECC3(x, edge_index, flow)
        # x = self.graphnorm3(x)
        # x = F.relu(x)

         #create graph embeddings
        embedding = global_mean_pool(x, batch)

        #prediciton head
        f_out = self.fcl1(embedding)
        f_out = F.relu(f_out)
        predicion = self.fcl2(f_out)

        return predicion
