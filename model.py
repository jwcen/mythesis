import torch
from torch import nn
from torch.autograd import Variable

import torch_geometric.nn as pyg_nn


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = pyg_nn.GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = (self.conv1(x, edge_index))
        self.feature = self.conv2(x, edge_index)
        return self.feature

class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(DKT, self).__init__()
        self.device = device
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim*2, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        # Readout layer
        self.fc = nn.Linear(hidden_dim+input_dim, output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, q_next):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
            
        # One time step
        out_next, hn = self.rnn(x, h0)
        out_next = torch.cat((out_next, q_next), axis=2)
#         out_next = torch.cat((out.reshape(-1,hidden_dim), q_next.reshape(-1,20)), axis=1)
        out = (self.fc(out_next))
        return out