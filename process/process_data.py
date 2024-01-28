import numpy as np
import pandas as pd
import os
import time 
import joblib
from tqdm import notebook, tqdm
from scipy.sparse import coo_matrix
from functools import partial
import multiprocessing

import sys
sys.path.append("..")

from coo_matrix import co_acc_probs, co_acc_skills




DATA_FOLDER = 'E:\\Cenjw\\研究生\\mythesis\\data\\{}'

def process(pool, dataset='assist09', encoding="ISO-8859-1"):
    data_folder = DATA_FOLDER.format(dataset)
    filepath = os.path.join(data_folder, f'{dataset}.csv')
    df = pd.read_csv(filepath, low_memory=False, encoding=encoding)

    if dataset == 'assist17':
        pass

    if dataset == 'matmat':
        ## 预处理数据集
        # df_a = pd.read_csv(f'{data_folder}/answers.csv', low_memory=False, encoding="ISO-8859-1")
        # # df_s = pd.read_csv(f'{data_folder}/skills.csv')
        # df_i = pd.read_csv(f'{data_folder}/items.csv')

        # df_i = df_i.rename(columns={'id': 'problem_id', 'skill':'skill_id'})
        # df_i = df_i[['problem_id',
        #         'question',	'visualization',
        #         'skill_id',	'skill_lvl_1',	'skill_lvl_2',	'skill_lvl_3',
        #         'data']]
        # df_a = df_a.rename(columns={'item': 'problem_id'})
        # df_merge = pd.merge(df_a,df_i,on='problem_id')
        # df_merge.to_csv(f'{data_folder}/matmat.csv', index=False)

        df['original'] = 1
        df = df.rename(columns={
            'student': 'user_id'
        })

    if dataset == 'ednet':
        df = df.rename(columns={
            'item_id': 'problem_id',
            'new_sort_skill_id': 'skill_id'
        })

        df = df.assign(skill_id=df.skill_id.str.split('-')).explode('skill_id')
        df['original'] = 1

    # delete empty skill_id
    df = df.dropna(subset=['skill_id'])
    df = df[~df['skill_id'].isin(['noskill'])]
    df.skill_id = df.skill_id.astype('int')
    print('After removing empty skill_id, records number %d' % len(df))

    # delete scaffolding problems
    df = df[df['original'].isin([1])]
    print('After removing scaffolding problems, records number %d' % len(df))

    min_inter_num = 3
    users = df.groupby(['user_id'], as_index=True)
    delete_users = []
    for u in users:
        if len(u[1]) < min_inter_num:
            delete_users.append(u[0])
    print('deleted user number based min-inters %d' % len(delete_users))
    df = df[~df['user_id'].isin(delete_users)]
    # df = df[[ 'user_id', 'problem_id', 'skill_id', 'correct']]
    print('After deleting some users, records number %d' % len(df))

    # 5,"[55969, 55970, ...
    skill_df = df[['skill_id', 'problem_id']].groupby(['skill_id'], as_index=True).apply(lambda r: np.array(list(set(r['problem_id'].values))))
    skill_df_path = os.path.join(data_folder, 'skill_prob.pkl.zip')
    joblib.dump(skill_df, skill_df_path)

    user_prob = df[['user_id', 'problem_id', 'correct']].groupby(['user_id', 'problem_id'])['correct'].agg('mean')

    processed_data = os.path.join(data_folder, f'processed_{dataset}.csv')
    df.to_csv(processed_data, index=False)
    print(f'Saved processed data to {processed_data}\n')

    # 抽取特征矩阵
    extract_feature_matrix(df, dataset, data_folder)

    # 抽取练习、知识点影响子图
    extract_subgraph(pool, df, skill_df, user_prob, data_folder)


def extract_feature_matrix(df, dataset, data_folder):
    
    if dataset == 'assist09':
        problems = df['problem_id'].unique()
        pro_id_dict = dict(zip(problems, range(len(problems))))
        print('problem number %d' % len(problems))

        pro_type = df['answer_type'].unique()
        pro_type_dict = dict(zip(pro_type, range(len(pro_type))))
        print('problem type: ', pro_type_dict)
        pro_feat = []
        for pro_id in range(len(problems)):
            tmp_df = df[df['problem_id'] == problems[pro_id]]
            tmp_df_0 = tmp_df.iloc[0]

            # pro_feature: [ms_of_response, answer_type, mean_correct_num]
            ms = tmp_df['ms_first_response'].abs().mean()
            p = tmp_df['correct'].mean()
            pro_type_id = pro_type_dict[tmp_df_0['answer_type']]
            tmp_pro_feat = [0.] * (len(pro_type_dict) + 2)
            tmp_pro_feat[0] = ms
            tmp_pro_feat[pro_type_id + 1] = 1.
            tmp_pro_feat[-1] = p
            pro_feat.append(tmp_pro_feat)
        
        pro_feat = np.array(pro_feat).astype(np.float32)
        pro_feat[:, 0] = (pro_feat[:, 0] - np.min(pro_feat[:, 0])) / \
            (np.max(pro_feat[:, 0]) - np.min(pro_feat[:, 0]))
        
        # save pro_feat_arr   
        pro_feat_path = os.path.join(data_folder, 'pro_feat.zip')
        joblib.dump(pro_feat, pro_feat_path)
        print(f'saved {dataset} pro_feat')
        # np.savez(f'{data_folder}/pro_feat.npz', pro_feat=pro_feat)

    if dataset == 'ednet':
        pass 

    if dataset == 'matmat':
        pass 

def extract_subgraph(pool, df, skill_df, user_prob, data_folder):
    skill_acc = df[['skill_id', 'correct']].groupby('skill_id')['correct'].agg('mean')
    user_skill = df[['user_id', 'skill_id', 'correct']].groupby(['user_id', 'skill_id'])['correct'].agg('mean')

    for user, skill in tqdm(user_skill.index):
        if user_skill[user][skill]>=skill_acc[skill]:
            user_skill[user][skill] = 1
        else:
            user_skill[user][skill] = 0

    # c2c matrix
    row = []
    col = []
    val = []
    print('Extracting c2c matrix...')
    arg_skills = ((skill1, skill2) for (i, skill1) in enumerate(skill_df.index) for skill2 in skill_df.index[i:])
    c2c = partial(co_acc_skills, user_skill)
    val = pool.starmap(c2c, arg_skills)
    for i in range(len(skill_df.index)):
        for j in range(i, len(skill_df.index)):
            row.append(i)
            col.append(j)
    mat_skill = coo_matrix((val, (row, col)), shape=(len(skill_df.index), len(skill_df.index)))

    joblib.dump(mat_skill, os.path.join(data_folder, 'c2c.pkl.zip'))
    print('processed c2c matrix\n')

    # e2e matrix
    print('Extracting e2e matrix...')
    skill_mats = []
    e2e = partial(co_acc_probs, user_prob)

    for skill in tqdm(skill_df.index):
        print('processing skill: ', skill)
        skill_probs = skill_df[skill]
        row = []
        col = []
        args = ((prob1, prob2) for i, prob1 in enumerate(skill_probs) for prob2 in skill_probs[i:])
        val = pool.starmap(e2e, args)
  
        for i in range(len(skill_probs)):
            for j in range(i, len(skill_probs)):
                row.append(i)
                col.append(j)
        
        assert(len(val)==len(row))
        mat = coo_matrix((val, (row, col)), shape=(len(skill_probs), len(skill_probs)))
        skill_mats.append(mat)

    joblib.dump(skill_mats,os.path.join(data_folder, 'para_e2e.pkl.zip'))
    print('processed e2e matrix\n')


if __name__ == '__main__':
    cores = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(processes=cores)

    dataset = 'assist09'
    print(f'start processing dataset {dataset}...')
    start = time.time()
    process(pool, dataset=dataset)
    end = time.time()
    print(f'total processed time: {end-start} s')