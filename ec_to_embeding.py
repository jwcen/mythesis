import imp
import torch
import joblib
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import train_test_split_edges

from model import Encoder

def train(epoch):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()

#     writer.add_scalar("loss", loss.item(), epoch)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


path = '/home/ktcodes/jktModel/data/a09'
a09c2c = '../data/a09/c2c.pkl.zip'
a09sp = '../data/a09/skill_prob.pkl.zip'

c2c = joblib.load(a09c2c)
skill_df = joblib.load(a09sp)

c2c_t = c2c.transpose()
c2c_t.setdiag(0)
c2c_add = c2c+c2c_t

# train c2c embedding
x = torch.tensor(c2c_add.toarray().tolist(), dtype=torch.float)
edge_index, edge_weight = pyg_utils.convert.from_scipy_sparse_matrix(c2c_add)
data = Data(edge_index=edge_index, x=x)


channels = 10
dev = torch.device('cuda:{1}' if torch.cuda.is_available() else 'cpu')
# device = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
print('CUDA availability:', torch.cuda.is_available())

model = pyg_nn.GAE(Encoder(c2c_add.shape[1], channels)).to(dev)
labels = data.y
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data, val_ratio=0, test_ratio=0.2)

x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 2001):
    train(epoch)
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    if epoch % 100 == 0:
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
c2c_emb = {}
for i, emb in enumerate(model.encoder.feature.cpu()):
    c2c_emb[skill_df.index[i]] = emb
# joblib.dump(c2c_emb, f'{path}/c2c_emb.pkl.zip') 


# train e2e embedding
e2e_mat = joblib.load(f'{path}/para_e2e.pkl.zip')
e2e_emb = {}
channels = 10
for i, e2e in enumerate(e2e_mat):
    skill_id = skill_df.index[i]
    if len(skill_df[skill_id]) >= channels:
        
        print('processing questions in skill: {} ======'.format(skill_id))
        e2e_t = e2e.transpose()
        e2e_t.setdiag(0)
        e2e_add = e2e + e2e_t

        x = torch.tensor(e2e_add.toarray().tolist(), dtype=torch.float)
        edge_index, edge_weight = pyg_utils.convert.from_scipy_sparse_matrix(e2e_add)
        data = Data(edge_index=edge_index, x=x)

        # encoder: written by us; decoder: default (inner product)
        model = pyg_nn.GAE(Encoder(e2e.shape[0], channels)).to(dev)
        # model = pyg_nn.GAE(Encoder(dataset.num_features, channels)).to(dev)

        labels = data.y
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        # data = model.split_edges(data)

        data = train_test_split_edges(data, val_ratio=0, test_ratio=0.2)

        x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        for epoch in range(1, 2001):
            train(epoch)
            auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
            # writer.add_scalar("AUC", auc, epoch)
            # writer.add_scalar("AP", ap, epoch)
            if epoch % 500 == 0:
                print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
        e2e_emb[skill_id] = model.encoder.feature.cpu()
# joblib.dump(e2e_emb, f'{path}/e2e_emb.pkl.zip')