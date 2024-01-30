from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KTDataset
from model import OurKT

import numpy as np 
import pandas as pd 
import torch
import joblib
import logging
import argparse
import time 
import os
import random 

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_embed_and_norm(params):
    e2e_emb = joblib.load(f'{params.data_folder}/e2e_emb.pkl.zip')
    c2c_emb = joblib.load(f'{params.data_folder}/c2c_emb.pkl.zip')
    skill_prob = joblib.load(f'{params.data_folder}/skill_prob.pkl.zip')

    filtered_skill_prob = {}
    channel = 10
    for i, skill_id in enumerate(skill_prob.index):
        if len(skill_prob[skill_id])>= channel:
            filtered_skill_prob[skill_id] = skill_prob[skill_id]
    joblib.dump(filtered_skill_prob, f'{params.data_folder}/filtered_skill_prob.pkl.zip')

    # normalization
    scaler = StandardScaler()
    all_c_v = []
    for k,v in c2c_emb.items():
        all_c_v.extend(list(v.numpy()))
    all_c_v = scaler.fit_transform(np.array(all_c_v).reshape(-1,1))
    all_c_v1 = {}
    for i, (k,v) in enumerate(c2c_emb.items()):
        all_c_v1[k] = all_c_v[i*10:(i+1)*10].reshape(-1,)
    all_e_v = {}
    for skill,qu_embs in e2e_emb.items():
        q_num = qu_embs.shape[0]
        temp_all_v = qu_embs.numpy().reshape(-1,)
        temp_all_v = scaler.fit_transform(np.array(temp_all_v).reshape(-1,1))
        all_e_v[skill] = temp_all_v.reshape(-1, channel)

    skill_emb = {}
    for skill in tqdm(filtered_skill_prob.keys()):
        temp_c = (np.array(all_c_v1[skill]))
        temp_e = np.array(np.mean(all_e_v[skill], axis=0))
        skill_emb[skill] = np.append(temp_c, temp_e)
    prob_emb = {}
    for skill in tqdm(filtered_skill_prob.keys()):
        for i, prob in enumerate(filtered_skill_prob[skill]):
            temp_c = (np.array(all_c_v1[skill]))
            temp_e = (np.array(all_e_v[skill][i]))
            new_emb = np.append(temp_c, temp_e)
            if prob in prob_emb.keys():
                prob_emb[prob] = np.row_stack((prob_emb[prob], new_emb)).squeeze()
    #             print(prob_emb[prob].shape)
            else: prob_emb[prob] = new_emb
    for prob in tqdm(prob_emb.keys()):
        if len(prob_emb[prob].shape) > 1:
            prob_emb[prob] = np.mean(prob_emb[prob], axis=0)
  
    return prob_emb, skill_emb, filtered_skill_prob


def split_train_test_data(params):
    def cat(a, b):
        return np.concatenate((a, b))
    
    prob_emb, skill_emb, filtered_skill_prob = load_embed_and_norm(params)

    filename = f'{params.data_folder}/processed_{params.dataset}.csv'
    df = pd.read_csv(filename, low_memory=False, encoding="ISO-8859-1")
    df = df[df['skill_id'].isin(filtered_skill_prob.keys())]
    df['e_emb'] = df['problem_id'].apply(lambda r: prob_emb[r])
    df['c_emb'] = df['skill_id'].apply(lambda r: skill_emb[r])
    df['ec_emb'] = df.apply(lambda row: cat(row['c_emb'], row['e_emb']), axis=1)

    col_name = 'ec_emb'

    embd_dim = np.array(df[col_name].tolist()).shape[1]

    group_c = df[['user_id', params.component, 'correct']].groupby('user_id').apply(lambda r: (np.array(r[col_name].tolist()).squeeze(), r['correct'].values))
    train_group_c = group_c.sample(frac=0.8, random_state=2023)
    test_group_c = group_c[~group_c.index.isin(train_group_c.index)]
    joblib.dump(train_group_c, f'{params.data_folder}/train_group_c.pkl.zip')
    joblib.dump(test_group_c, f'{params.data_folder}/test_group_c.pkl.zip')

    return embd_dim, train_group_c, test_group_c


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='assist09', help='路径或名称指定数据集')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=200, help='number of iterations')
    parser.add_argument('--lr', type=float, default= 0.001, help='initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1E-3, help='learning rate will not decrease after hitting this threshold')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--input_dim', type=int, default=20, help='embedding dimensions')
    parser.add_argument('--hidden_dim', type=float, default=128, help='hidden state dim for LSTM')
    parser.add_argument('--layer_dim', type=float, default=2, help='layer number for LSTM')
    parser.add_argument('--output_dim', type=float, default=1, help='layer number for LSTM')
    parser.add_argument('--max_step', type=int, default=50, help='the allowed maximum length of a sequence')
    parser.add_argument('--fold', type=str, default='1', help='number of fold')
    parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
    parser.add_argument('--component', type=str, default='c_emb', help='component')
    params = parser.parse_args()
    params.data_folder = f'/home/cen_jianwei/code/AttGAE-KT/data/{params.dataset}'

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_filepath = f'{params.data_folder}/log/prediction'
    if not os.path.exists(log_filepath):
        os.makedirs(log_filepath)
    fh = logging.FileHandler(
            f'{params.data_folder}/log/{str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))}.log'
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
    logger.info(f"input dim: {params.input_dim}")
    logger.info(f"hidden dim: {params.hidden_dim}")
    logger.info(f"layer dim: {params.layer_dim}")
    logger.info(f"output dim: {params.output_dim}")
    logger.info(f"lr: {params.lr}")
    logger.info(f"max_lr: {params.max_lr}")
    logger.info(f"max iter: {params.max_iter}")
    logger.info(f"batch_size: {params.batch_size}")
    logger.info(f"max step: {params.max_step}")


    embed_dim, train_group_c, test_group_c = split_train_test_data(params)
    
    train_dataset = KTDataset(train_group_c, max_seq=params.max_step)
    train_dataloader  = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    valid_dataset = KTDataset(test_group_c, max_seq=params.max_step)
    valid_dataloader  = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=True)


    device = torch.device(
        f"cuda:{params.gpu}" if torch.cuda.is_available() and params.gpu >= 0 else "cpu")

    logger.info(f'embed dim: {embed_dim}')
    model = OurKT(
                    embed_dim
                  , params.hidden_dim
                  , params.layer_dim
                  , params.output_dim
                  , device
                  , params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=params.max_lr,
        steps_per_epoch=len(train_dataloader),
        epochs=params.max_iter,
        verbose=True
    )

    model.to(device)
    criterion.to(device)

    best_acc, best_auc = 0, 0
    for epoch in (range(params.max_iter)):
        train_loss, acc, auc = model.train_fn(train_dataloader, optimizer, criterion, scheduler=scheduler)
        loss, acc, pre, rec, f1, auc = model.valid_fn(valid_dataloader, criterion)
        res = "epoch - {}/{} train loss:{:.4f}, test acc: {:.4f}, test auc: {:.4f}".format(epoch+1, params.max_iter, train_loss, acc, auc)
        # res = "epoch - {}/{} train loss - {:.3f}, test acc - {:.3f}, test auc - {:.3f}, rec - {:.3f}, f1 - {:3f}, pre - {:.3f}".format(epoch+1, params.max_iter, loss, acc, auc, rec, f1, pre)
        logger.info(res)

        best_acc = max(best_acc, acc)
        best_auc = max(best_auc, auc)

    logger.info(f'best auc: {best_auc}, best acc: {best_acc}')


if __name__ == "__main__":
    run()