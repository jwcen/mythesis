import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch import nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, precision_score, f1_score, recall_score


class GATEncoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, heads=2, alpha=0.2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.alpha = alpha
        
        self.conv = pyg_nn.GATConv(in_channels, out_channels, heads)
        
        self.bn = nn.BatchNorm1d(out_channels * heads)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        
        x = self.conv(x, edge_index) 
        x = self.bn(x)
        x = self.dropout(x)
        self.x = self.relu(x)
        
        return self.x


class OurKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device, params):
        super(OurKT, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.params = params
        self.lstm = nn.LSTM(input_dim*2
                           , hidden_dim
                           , layer_dim
                           , batch_first=True
                        #    , nonlinearity='tanh'
                            , dropout=params.dropout 
                           , device=self.device)
        self.fc = nn.Linear(hidden_dim + input_dim, hidden_dim, bias=True, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True, device=self.device)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        
    def forward(self, x, q_next):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        
        # One time step
        # print('x: \n', type(x), x.shape)  #  <class 'torch.Tensor'> torch.Size([128, 49, 60])
        out_next, (hn, cn) = self.lstm(x, (h0, c0))
        out_next = torch.cat((out_next, q_next), axis=2)
#         out_next = torch.cat((out.reshape(-1,hidden_dim), q_next.reshape(-1,20)), axis=1)
        # y = self.relu(self.fc(out_next))
        y = self.relu(self.fc(out_next))
        pred = (self.fc2(y))
        return pred
    
    def train_fn(self, dataloader, optimizer, criterion, scheduler=None):
        self.train()

        train_loss = []
        num_corrects = 0
        num_total = 0
        labels = []
        outs = []

        for x_emb, q_next, y in (dataloader):
            x = x_emb.to(self.device).float()
            y = y.to(self.device).float()
            q_next = q_next.to(self.device).float()

            out = self.forward(x, q_next).squeeze()

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            target_mask = (q_next!=0).unique(dim=2).squeeze()

            filtered_out = torch.masked_select(out, target_mask)
            filtered_label = torch.masked_select(y, target_mask)
            filtered_pred = (torch.sigmoid(filtered_out) >= 0.5).long()

            num_corrects += (filtered_pred == filtered_label).sum().item()
            num_total += len(filtered_label)

            labels.extend(filtered_label.view(-1).data.cpu().numpy())
            outs.extend(filtered_pred.view(-1).data.cpu().numpy())

        acc = num_corrects / num_total
        auc = roc_auc_score(labels, outs)
        loss = np.mean(train_loss)

        return loss, acc, auc
    
    def valid_fn(self, dataloader, criterion):
        self.eval()

        valid_loss = []
        num_corrects = 0
        num_total = 0
        labels = []
        outs = []

        for x_emb, q_next, y in (dataloader):
            x = x_emb.to(self.device).float()
            y = y.to(self.device).float()
            q_next = q_next.to(self.device).float()
            out = self.forward(x, q_next).squeeze()

            loss = criterion(out, y)
            valid_loss.append(loss.item())

            target_mask = (q_next!=0).unique(dim=2).squeeze()

            filtered_out = torch.masked_select(out, target_mask)
            filtered_label = torch.masked_select(y, target_mask)
            filtered_pred = (torch.sigmoid(filtered_out) >= 0.5).long()

            num_corrects += (filtered_pred == filtered_label).sum().item()
            num_total += len(filtered_label)

            labels.extend(filtered_label.view(-1).data.cpu().numpy())
            outs.extend(filtered_pred.view(-1).data.cpu().numpy())

        acc = num_corrects / num_total
        auc = roc_auc_score(labels, outs)
        pre = precision_score(labels, outs, zero_division=1)
        f1 = f1_score(labels, outs, zero_division=1)
        rec = recall_score(labels, outs, zero_division=1)
        loss = np.mean(valid_loss)

        return loss, acc, pre, rec, f1, auc
