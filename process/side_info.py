from concurrent.futures import process

import os
import pandas as pd
import numpy as np
from scipy import sparse



def process_csv(filename, min_inter_num, data_folder='assist09'):
    """
    pre-process original csv file for assist dataset
    """

    # read csv file
    data_path = '/home/cenjw/kt/PEBG-master/assist09/{}'.format(filename);
    df = pd.read_csv(data_path, low_memory=False, encoding="ISO-8859-1")
    print('original records number %d' % len(df))

    # delete empty skill_id
    df = df.dropna(subset=['skill_id'])
    df = df[~df['skill_id'].isin(['noskill'])]
    print('After removing empty skill_id, records number %d' % len(df))

    # delete scaffolding problems
    df = df[df['original'].isin([1])]
    print('After removing scaffolding problems, records number %d' % len(df))

    # delete the users whose interaction number is less than min_inter_num
    users = df.groupby(['user_id'], as_index=True)
    delete_users = []
    for u in users:
        if len(u[1]) < min_inter_num:
            delete_users.append(u[0])
    print('deleted user number based min-inters %d' % len(delete_users))
    df = df[~df['user_id'].isin(delete_users)]
    print('After deleting some users, records number %d' % len(df))
    # print('features: ', df['assistment_id'].unique(), df['answer_type'].unique())

    df.to_csv('%s_processed.csv' % data_folder)

def get_features():
    df = pd.read_csv('assist09_processed.csv',
                     low_memory=False, encoding="ISO-8859-1")
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
    import joblib
    joblib.dump(pro_feat, './assist09_pro_feat.zip')
    np.savez('./assist09_pro_feat.npz', pro_feat=pro_feat)


if __name__ == '__main__':
    
    min_inter_num = 3
    file_name = 'skill_builder_data_corrected_collapsed.csv'
    # process_csv(file_name, min_inter_num)
    get_features()