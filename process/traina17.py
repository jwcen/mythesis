import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
from model import DKT
from myutils import train_fn, valid_fn
from data.Dataset import DKTDataset
import time
import argparse
import sys


### Argument and global variables
parser = argparse.ArgumentParser(
    'Interface for TGAT experiments on link predictions')
parser.add_argument('--bs', type=int, default=64, help='batch_size')
parser.add_argument('--n_epoch', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1,
                    help='dropout probability')
parser.add_argument('--gpu', type=int, default=0,
                    help='idx for the gpu to use')
parser.add_argument('--input_dim', type=int, default=20,
                    help='')
parser.add_argument('--layer_dim', type=int, default=1,
help='')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='')
parser.add_argument('--output_dim', type=int, default=1,
                    help='')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
# Model settings
input_dim = args.input_dim  # input dimension
hidden_dim = 100  # hidden layer dimension
layer_dim = 1  # number of hidden layers,default=1
output_dim = args.output_dim  # output dimension
MAX_LEARNING_RATE = 2e-3
LEARNING_RATE = 3e-3
EPOCHS = 1001
BATCH_SIZE = 128
MAX_SEQ = 50
M_NAME = "GAT"
DATASET_NAME = 'a17'


# set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(
    './log/{}-{}.log'.format(DATASET_NAME +"-"+M_NAME, str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


logger.info("Encode model: " + M_NAME)

data_path = './data/a17/emb'
logger.info("dataset directory: "+data_path)

logger.info("input dim: " + str(input_dim))
logger.info("hidden dim: " + str(hidden_dim))
logger.info("layer dim: " + str(layer_dim))
logger.info("output dim: " + str(output_dim))
logger.info("lr: "+ str(LEARNING_RATE))
logger.info("max_lr: " + str(MAX_LEARNING_RATE))
logger.info("epochs: "+ str(EPOCHS))
logger.info("batch_size: " + str(BATCH_SIZE))



e2e_emb = joblib.load(f'{data_path}/e2e_emb.pkl.zip')
c2c_emb = joblib.load(f'{data_path}/c2c_emb.pkl.zip')
skill_prob = joblib.load('./data/a17/skill_prob.pkl.zip')

filtered_skill_prob = {}
channel = 10
for i, skill_id in enumerate(skill_prob.index):
    if len(skill_prob[skill_id]) >= channel:
        filtered_skill_prob[skill_id] = skill_prob[skill_id]
joblib.dump(filtered_skill_prob, f'{data_path}/filtered_skill_prob.pkl.zip')

# normalization
scaler = StandardScaler()
all_c_v = []
for k, v in c2c_emb.items():
    all_c_v.extend(list(v.numpy()))
all_c_v = scaler.fit_transform(np.array(all_c_v).reshape(-1, 1))
all_c_v1 = {}
for i, (k, v) in enumerate(c2c_emb.items()):
    all_c_v1[k] = all_c_v[i*10:(i+1)*10].reshape(-1,)
all_e_v = {}
for skill, qu_embs in e2e_emb.items():
    q_num = qu_embs.shape[0]
    temp_all_v = qu_embs.numpy().reshape(-1,)
    temp_all_v = scaler.fit_transform(np.array(temp_all_v).reshape(-1, 1))
    all_e_v[skill] = temp_all_v.reshape(-1, 10)


skill_emb = {}
for skill in tqdm(filtered_skill_prob.keys()):
    # print(skill)
    if skill in all_c_v1:
        temp_c = (np.array(all_c_v1[skill]))
    if skill in all_e_v:
        temp_e = np.array(np.mean(all_e_v[skill], axis=0))
    skill_emb[skill] = np.append(temp_c, temp_e)
prob_emb = {}
for skill in tqdm(filtered_skill_prob.keys()):
    for i, prob in enumerate(filtered_skill_prob[skill]):
        if skill in all_c_v1:
            temp_c = (np.array(all_c_v1[skill]))
        if skill in all_e_v:
            temp_e = (np.array(all_e_v[skill][i]))
        new_emb = np.append(temp_c, temp_e)
        if prob in prob_emb.keys():
            prob_emb[prob] = np.row_stack((prob_emb[prob], new_emb)).squeeze()
#             print(prob_emb[prob].shape)
        else:
            prob_emb[prob] = new_emb
for prob in tqdm(prob_emb.keys()):
    if len(prob_emb[prob].shape) > 1:
        prob_emb[prob] = np.mean(prob_emb[prob], axis=0)


# Train/Test data
# read in the data
print('read in the datasete...')
df = pd.read_csv("./data/a17/student_log_2.csv")
df = df[~df['skill'].isin(['noskill'])]
df['skill_id'], _ = pd.factorize(df['skill'], sort=False)
df.skill_id = df['skill_id'] + 1
# df['problem_id'], _ = pd.factorize(df['problemId'], sort=False)
df.rename(columns={"ITEST_id": "user_id",
          "problemId": "problem_id"}, inplace=True)
        
# delete scaffolding problems
df = df[df['original'].isin([1])]
print('After removing scaffolding problems, records number %d' % len(df))

min_inter_num = 3
users = df.groupby(['user_id'], as_index=True)
delete_users = []
for u in users:
    if len(u[1]) < min_inter_num:
        delete_users.append(u[0])
logger.info('deleted user number based min-inters %d' % len(delete_users))
df = df[~df['user_id'].isin(delete_users)]
df = df[[ 'user_id', 'problem_id', 'skill_id', 'correct']]
logger.info('After deleting some users, records number %d' % len(df))

df = df[df['skill_id'].isin(filtered_skill_prob.keys())]
df['skill_cat'] = df['skill_id'].astype('category').cat.codes
df['e_emb'] = df['problem_id'].apply(lambda r: prob_emb[r])
df['c_emb'] = df['skill_id'].apply(lambda r: skill_emb[r])

feats = joblib.load('./data/a17/feats.pkl.zip')

df['feats'] = df['skill_id'].apply(lambda r: feats[r])

def cat(a, b):
    return np.concatenate((a, b))


print('++++++')
# print(emb_dim)
df['ec_emb'] = df.apply(lambda row: cat(row['c_emb'], row['e_emb']), axis=1)
df['ecf_emb'] = df.apply(lambda row: cat(row['ec_emb'], row['feats']), axis=1) 

emb_dim = np.array(df['ecf_emb'].tolist()).shape[1]

group_c = df[['user_id', 'ecf_emb', 'correct']].groupby('user_id').apply(
    lambda r: (np.array(r['ecf_emb'].tolist()).squeeze(), r['correct'].values))
train_group_c = group_c.sample(frac=0.8, random_state=2020)
test_group_c = group_c[~group_c.index.isin(train_group_c.index)]
joblib.dump(train_group_c, './data/a17/train_group_c.pkl.zip')
joblib.dump(test_group_c, './data/a17/test_group_c.pkl.zip')


train_dataset = DKTDataset(train_group_c, max_seq=MAX_SEQ)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataset = DKTDataset(test_group_c, max_seq=MAX_SEQ)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

model = DKT(emb_dim, hidden_dim, layer_dim, output_dim, device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.RAdam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
# )

model.to(device)
criterion.to(device)



if __name__ == '__main__':
    train_acc_list = []
    train_auc_list = []
    valid_acc_list = []
    valid_auc_list = []
    train_loss_list = []
    valid_loss_list = []

    import time

    for epoch in range(EPOCHS):
        loss, acc, auc = train_fn(
            model, train_dataloader, optimizer, criterion, device)
        logger.info("epoch: {}/{} train => loss: {:.3f}, acc: {:.3f}, auc: {:.3f}".format(
            epoch+1, EPOCHS, loss, acc, auc))

        train_acc_list.append(acc)
        train_auc_list.append(auc)
        train_loss_list.append(loss)
        

        loss, acc, pre, rec, f1, auc = valid_fn(
            model, valid_dataloader, criterion, device)
        res = "epoch: {}/{} valid => loss: {:.3f}, acc: {:.3f}, auc: {:.3f}, pre: {:.3f}, f1: {:3f}, rec: {:.3f}".format(epoch + 1, EPOCHS, loss, acc, auc, pre, f1, rec)
        valid_acc_list.append(acc)
        valid_auc_list.append(auc)
        valid_loss_list.append(loss)
        logger.info(res)
    
    # logger.info("train loss: {}".format(train_loss_list))
    # logger.info("valid loss: {}".format(valid_loss_list))
    # logger.info("train accs: {}".format(train_acc_list))
    # logger.info("valid accs: {}".format(valid_acc_list))
    # logger.info("train aucs: {}".format(train_auc_list))
    # logger.info("valid_aucs: {}".format(valid_auc_list))