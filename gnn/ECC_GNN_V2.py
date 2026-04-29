import torch.nn as nn
from torch_geometric.nn import NNConv,  GraphNorm, global_mean_pool
import torch.nn.functional as F

class MCC_GNN (nn.Module):

    def __init__(self, edge_dimension, node_dimension, hidden_dimension):
        super(MCC_GNN, self).__init__()


       #pass 1 of ecc
        edge_weights1 = nn.Sequential(
            nn.Linear(edge_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, node_dimension * hidden_dimension) #shape in_channels * out_channels needed for nnconv

        )
        #ecc, output is of dimension hidden dimensits
        self.ECC1 = NNConv(node_dimension, hidden_dimension, edge_weights1, aggr= 'mean') #aggr mean to prevnt hubs from dominating

        #graph norm pass 1
        self.graphnorm1 = GraphNorm(hidden_dimension)


        #pass 2 of ecc
        edge_weights2 = nn.Sequential(
            nn.Linear(edge_dimension , hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension * hidden_dimension)

        )
        #ecc
        self.ECC2 = NNConv(hidden_dimension, hidden_dimension, edge_weights2,
                           aggr='mean') #in channels is output from pass 1, so hidden dim
        # graph norm pass 2
        self.graphnorm2 = GraphNorm(hidden_dimension)


        #fully connected layer
        self.fcl1 = nn.Linear(hidden_dimension, hidden_dimension //2)
        self.fcl2 =nn.Linear(hidden_dimension // 2 ,1)


    def forward(self, data):

        node_features = data.x #popullation, area , position
        edge_index =  data.edge_index
        flow = data.edge_attr #mobility flow

        batch = data.batch

         #pass 1
        x = self.ECC1(node_features, edge_index, flow)
        x = F.relu(x) #order from paper
        x = self.graphnorm1(x)

        #pass 2
        x = self.ECC2(x, edge_index, flow)
        x = F.relu(x)
        x = self.graphnorm2(x)

        #relu here,  cite paper
        x = F.relu(x)
         #create graph embeddings
        embedding = global_mean_pool(x, batch)

        #prediciton head
        f_out = self.fcl1(embedding)
        f_out = F.relu(f_out)
        predicion = self.fcl2(f_out)





        return predicion
