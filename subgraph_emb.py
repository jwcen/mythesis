import os 
import warnings

import torch
import joblib
import torch.nn as nn 
import time, joblib
import torch.nn.functional as F

import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score

from model import GATEncoder


warnings.simplefilter("ignore")


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)

    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = model.decoder(z, pos_edge_index, sigmoid=True)
    neg_pred = model.decoder(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()  
    auc = roc_auc_score(y, pred)
    ap = average_precision_score(y, pred)

    return auc, ap

################# start ####################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASE_DIR', type=str, default='/home/cen_jianwei/code/AttGAE-KT', help='模型所在根目录')
    parser.add_argument('--dataset', type=str, default='assist09')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=-1)

    params = parser.parse_args()

    # load data
    DATA_FOLDER = f'{params.BASE_DIR}/data/{params.dataset}'  # 学校服务器
    
    saved_model_folder = os.path.join(params.BASE_DIR, 'AttGAE-KT')
    if not os.path.exists(saved_model_folder):
        os.mkdir(saved_model_folder)

    # set up logger
    import logging 
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_filepath = f'{DATA_FOLDER}/log/pre-train'
    if not os.path.exists(log_filepath):
        os.makedirs(log_filepath)
    fh = logging.FileHandler(
            f'{DATA_FOLDER}/log/{str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))}.log'
        )
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"hidden dim: {params.hidden_dim}")
    logger.info(f"heads: {params.heads}")
    logger.info(f"channels: {params.channels}")
    logger.info(f"lr: {params.lr}")
    logger.info(f"alpha: {params.alpha}")
    logger.info(f"epochs: {params.epochs}")
    logger.info(f"epochs: {params.dropout}")

    if params.gpu >= 0 and torch.cuda.is_available():
        dev = torch.device(f'cuda:{params.gpu}')
    else:
        dev = torch.device('cpu')

    #  (0, 0)        0.782608695652174
    #   (0, 1)        0.6213592233009708
    # ...
    c2c_coo = joblib.load(os.path.join(DATA_FOLDER, 'c2c.pkl.zip'))
    skill_prob = f'{DATA_FOLDER}/skill_prob.pkl.zip'
    # 2  [194, 555]
    skill_df = joblib.load(skill_prob)


    print('====train c2c embedding====')
    c2c_t = c2c_coo.transpose()  # (0, 1) => (1, 0)
    c2c_t.setdiag(0)
    c2c_add = c2c_coo + c2c_t

    # tensor([[0.7826, 0.6214, 1.0000,  ...,]
    x = torch.tensor(c2c_add.toarray().tolist(), dtype=torch.float)
    
    # index tensor([[  0,   0,   0,  ..., 101, 101, 101],
    #         [  0,   1,   2,  ...,  99, 100, 101]])
    # weight tensor([0.7826, 0.6214, 1.0000,  ..., 0.3667, 0.3777, 0.5417],
    #        dtype=torch.float64)
    edge_index, edge_weight = pyg_utils.convert.from_scipy_sparse_matrix(c2c_add)
    
    # Data(x=[102, 102], edge_index=[2, 9946])
    data = Data(edge_index=edge_index, x=x)  
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data, val_ratio=0, test_ratio=0.2)
    x = data.x.to(dev)
    train_pos_edge_index = data.train_pos_edge_index.to(dev)

    model = pyg_nn.GAE(GATEncoder(c2c_add.shape[1], params.channels)).to(dev)
    optimizer = torch.optim.Adam(model.parameters() , lr=params.lr)

    for epoch in range(1, (params.epochs + 1)):
        loss = train()
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        if epoch % 100 == 0:
            logger.info('Epoch: {:03d}, Train loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))

        
    c2c_emb = {}
    for i, emb in enumerate(model.encoder.x.cpu()):
        if i < len(skill_df.index):
            c2c_emb[skill_df.index[i]] = emb
    
    joblib.dump(c2c_emb, f'{DATA_FOLDER}/c2c_emb.pkl.zip') 
    print('saved c2c_emb')


    time.sleep(1)
    print('==========e2e embedding===========')

    # [<2x2 sparse matrix of type '<class 'numpy.float64'>'...]
    e2e_mats = joblib.load(os.path.join(DATA_FOLDER, 'para_e2e.pkl.zip')) 
    e2e_emb = {}

    for i, e2e in enumerate(e2e_mats):
        skill_id = skill_df.index[i]
        if len(skill_df[skill_id]) >= params.channels:
            print('processing questions in skill: {} ======'.format(skill_id))
            e2e_t = e2e.transpose()
            e2e_t.setdiag(0)
            e2e_add = e2e + e2e_t

            x = torch.tensor(e2e_add.toarray().tolist(), dtype=torch.float)
            edge_index, edge_weight = pyg_utils.convert.from_scipy_sparse_matrix(e2e_add)
            data = Data(edge_index=edge_index, x=x)
            data.train_mask = data.val_mask = data.test_mask = data.y = None
            data = train_test_split_edges(data, val_ratio=0, test_ratio=0.2)
            
            x = data.x.to(dev)
            train_pos_edge_index = data.train_pos_edge_index.to(dev)
            
            model = pyg_nn.GAE(GATEncoder(e2e.shape[0], params.channels)).to(dev)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

            for epoch in range(1, (params.epochs + 1)):
                loss = train()
                auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
                if epoch % 100 == 0:
                    logger.info('Epoch: {:03d}, Train loss:{:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))
                
            e2e_emb[skill_id] = model.encoder.x.cpu()

    joblib.dump(e2e_emb, f'{DATA_FOLDER}/e2e_emb.pkl.zip')
    print('e2e embeding saved!')